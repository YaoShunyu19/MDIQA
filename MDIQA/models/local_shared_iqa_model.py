import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from pyiqa.metrics import calculate_metric
from pyiqa.utils import get_root_logger, imwrite, tensor2img
from .base_model import BaseModel
import matplotlib.pyplot as plt
import os

from MDIQA.archs import build_network
from pyiqa.archs import build_network as build_network_base
from MDIQA.losses import build_loss
from MDIQA.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LocalSharedIQAModel(BaseModel):
    """General module to train an IQA network."""

    def __init__(self, opt):
        super(LocalSharedIQAModel, self).__init__(opt)

        # predefine if dataset requires score range
        self.dataset2headidx, self.dataset2scorerange = {}, {}
        for k, v in opt['dataset_head_idx_mapping'].items():
            self.dataset2headidx[k] = v
        for _, v in opt['datasets'].items():
            if 'score_range' in v.keys():
                self.dataset2scorerange[v['name']] = [float(v['score_range'][0]), float(v['score_range'][1])]

        # define network
        self.net_curr = build_network(opt['network'])
        self.net_curr = self.model_to_device(self.net_curr)
        self.net_bases, self.dataset2metric, self.lower_better = {}, {}, {}
        for _, v in opt['network_base'].items():
            metric = v['metric']
            net_base = build_network_base(metric)
            net_base = self.model_to_device(net_base)
            self.net_bases[metric['type']] = net_base
            self.lower_better[metric['type']] = v['lower_better']
            datasets = v['datasets']
            for d in datasets.values():
                self.dataset2metric[d] = metric['type']
        self.print_network(self.net_curr)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_curr, load_path, self.opt['path'].get('strict_load', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_curr.train()
        for _, net_base in self.net_bases.items():
            net_base.eval()
        train_opt = self.opt['train']

        self.net_best = build_network(self.opt['network']).to(self.device)

        # define losses
        self.dataset2loss = {}  # {'dataset': [{'type', 'cri'}, {} ... ], '': []}
        for k, v in train_opt['losses'].items():
            datasets = v['datasets']
            for d in datasets.values():
                cri = build_loss(v['opt']).to(self.device)
                loss = {'type': v['opt']['type'], 'cri': cri}
                if d in self.dataset2loss.keys():
                    self.dataset2loss[d].append(loss)
                else:
                    self.dataset2loss[d] = [loss]

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.train_backbone_iter = self.opt['train'].get('train_backbone_iter', -1)

        self.l_total = 0
        self.loss_dict = OrderedDict()

    def setup_optimizers(self, init=True):
        train_opt = self.opt['train']
        optim_opt = train_opt['optim']

        param_dict = {k: v for k, v in self.net_curr.named_parameters()}
        param_keys = list(param_dict.keys())
        # set different lr for different modules if needed, e.g., lr_backbone, lr_head
        lr_keys = [i for i in optim_opt.keys() if i.startswith('lr_')]

        optim_params = []
        for key in lr_keys:
            if key.startswith('lr_'):
                module_key = key.replace('lr_', '')
                logger = get_root_logger()
                logger.info(f'Set optimizer for {module_key} with lr: {optim_opt[key]}, weight_decay: {optim_opt.get(f"weight_decay_{module_key}", 0.)}')

                optim_params.append({
                    'params': [param_dict[k] for k in param_keys if module_key in k and param_dict[k].requires_grad],
                    'lr': optim_opt.pop(key, 0.),
                    'weight_decay': optim_opt.pop(f'weight_decay_{module_key}', 0.),
                })

                # should use param_keys[:] to avoid iteration error
                for k in param_keys[:]:
                    if module_key in k:
                        param_keys.remove(k)
        
        # append the rest of the parameters
        optim_params.append({
            'params': [param_dict[k] for k in param_keys if param_dict[k].requires_grad],
        })

        # log params that will not be optimized
        for k, v in param_dict.items():
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        
        # remove blank param list
        for k in optim_params:
            if len(k['params']) == 0:
                optim_params.remove(k)

        optim_type = train_opt['optim']['type']
        optim_config = {key: value for key, value in train_opt['optim'].items() if key != 'type'}
        self.optimizer = self.get_optimizer(optim_type, optim_params, **optim_config)

        if not init:
            self.optimizers.pop()
            self.schedulers[-1].optimizer = self.optimizer
            # for optimizer, scheduler in zip(self.optimizers, self.schedulers):
            #     scheduler.optimizer = optimizer
        
        self.optimizers.append(self.optimizer)

    def net_forward(self, net):
        return net(self.dist)
    
    def custom_feed_data(self, data, dataset_name):
        if dataset_name.startswith('koniq10k'):
            self.dist = data['img'].to(self.device)  # B,3,H,W
        else:
            self.dist = data['dist'].to(self.device)  # B,3,H,W

        headidx = self.dataset2headidx[dataset_name]
        global_score, local_score = self.net_curr(self.dist, headidx)  # B,1  B,d,h,w
        if global_score is not None:
            score_curr1, score_curr2 = score_dim_process(global_score)  # B  B,1
        
        # process each dataset
        if dataset_name.startswith('SPAQ') or dataset_name.startswith('PARA'):
            base_score = data['label'].to(self.device).float()  # B
            base_score1, base_score2 = score_dim_process(base_score)

        # calculate loss for each dataset
        batch_loss = 0.
        for loss in self.dataset2loss[dataset_name]:
            if loss['type'] == 'FidelityLoss':
                l = loss['cri'](score_curr1, base_score1)
            elif loss['type'] == 'NiNLoss':
                l = loss['cri'](score_curr2, base_score2)
            elif loss['type'] == 'MSELoss':
                l = loss['cri'](score_curr2, base_score2)
            batch_loss += l
            self.l_total += l
            if loss['type'] in self.loss_dict.keys():
                self.loss_dict[loss['type']] += l
            else:
                self.loss_dict[loss['type']] = l
        batch_loss.backward()

    def set_optimizer_zero_grad(self):
        self.optimizer.zero_grad()
    
    def optimize_parameters(self, current_iter):
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(self.loss_dict)

        self.l_total = 0
        self.loss_dict = OrderedDict()

        if current_iter == self.train_backbone_iter:
            self.net_curr.freeze_backbone()
            self.setup_optimizers(False)
            self.copy_model(self.net_curr, self.net_best)
            self.save_network(self.net_best.get_backbone(), f'net_freeze_backbone')

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        
        self.net_curr.eval()
        dataset_with_metrics = False

        # supervised with existed score
        headidx = self.dataset2headidx[dataset_name]
        if not isinstance(headidx, list):
            headidx = [headidx]
        preds_list = [[] for _ in range(len(headidx))]
        if dataset_name.startswith('SPAQ') or dataset_name.startswith('PARA') or dataset_name.startswith('koniq10k'):
            path_key = 'img_path' if dataset_name.startswith('koniq10k') else 'dist_path'
            img_key = 'img' if dataset_name.startswith('koniq10k') else 'dist'
            label_key = 'mos_label' if dataset_name.startswith('koniq10k') else 'label'
            dataset_with_metrics = True
            labels = []
            for _, val_data in enumerate(dataloader):
                img_path = val_data[path_key][0]
                img_name = osp.basename(img_path)
                self.dist = val_data[img_key].to(self.device)
                global_score_list, local_score_list = self.net_curr(self.dist)  # B,1  B,d,h,w
        
                label = val_data[label_key]
                label1, label2 = score_dim_process(label)
                for i in range(len(headidx)):
                    preds_list[i].append(global_score_list[headidx[i]])
                labels.append(label2)
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Test on {dataset_name} {img_name:>20}')
            if use_pbar:
                pbar.close()

        if with_metrics and dataset_with_metrics:
            preds_list = [torch.cat(preds, dim=0).squeeze(1).cpu().numpy() for preds in preds_list]
            labels = torch.cat(labels, dim=0).squeeze(1).cpu().numpy()
            
            save_dir = os.path.join(self.opt['path']['experiments_root'], 'figures', f'iter{current_iter}')
            os.makedirs(save_dir, exist_ok=True)

            for i, preds in enumerate(preds_list):
                idx = headidx[i]
                plt.cla()
                plt.scatter(preds, labels)
                # calculate all metrics
                num = 1
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] = abs(calculate_metric([preds, labels], opt_))
                    plt.text(0.05, 1.0 - 0.05 * num, f'{name}: {self.metric_results[name]:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                    num += 1
                save_filename = f'{dataset_name}_head{idx}.png'
                save_path = os.path.join(save_dir, save_filename)
                plt.savefig(save_path)
    
                if self.key_metric is not None:
                    # If the best metric is updated, update and save best model
                    if self.train_backbone_iter < 0:
                        to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                                    self.metric_results[self.key_metric], current_iter)
                        if to_update:
                            for name, opt_ in self.opt['val']['metrics'].items():
                                self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                            self.copy_model(self.net_curr, self.net_best)
                            self.save_network(self.net_best, f'net_best_{dataset_name}')
                            # self.save_network(self.net_best.get_heads(idx), f'net_best_{dataset_name}_head{idx}')
                    else:
                        if current_iter >= self.train_backbone_iter:
                            to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                                        self.metric_results[self.key_metric], current_iter)
                            if to_update:
                                for name, opt_ in self.opt['val']['metrics'].items():
                                    self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                                self.copy_model(self.net_curr, self.net_best)
                                self.save_network(self.net_best.get_heads(idx), f'net_best_{dataset_name}_head{idx}')

                self._log_validation_metric_values(current_iter, dataset_name, idx, tb_logger)

        self.net_curr.train()

    def _log_validation_metric_values(self, current_iter, dataset_name, headidx, tb_logger):
        log_str = f'Validation {dataset_name} head{headidx}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'val_metrics/{dataset_name}/{metric}', value, current_iter)


def score_dim_process(score):
    if score.ndim == 1:
        return score, score[:, None]
    elif score.ndim == 2:
        return score[:, 0], score