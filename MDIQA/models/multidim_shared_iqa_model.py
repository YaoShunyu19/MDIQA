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
class MultiDimSharedIQAModel(BaseModel):
    """General module to train an IQA network."""

    def __init__(self, opt):
        super(MultiDimSharedIQAModel, self).__init__(opt)
        logger = get_root_logger()

        self.update_until_better = opt.get('update_until_better', False)

        self.datasets_idx, idx = {}, 1
        for k, v in opt['datasets'].items():
            if v['name'] not in self.datasets_idx.keys():
                self.datasets_idx[v['name']] = idx
                idx += 1
        logger.info(f'Datasets idx:\n{self.datasets_idx}')

        if opt['network'].get('dataset_specific_weighted', False):
            self.dataset_specific_weighted = True
            opt['network']['dataset_specific_weighted'] = len(self.datasets_idx.keys())
            logger.info('Use Dataset Specific Weighted.')
        else:
            self.dataset_specific_weighted = False

        # define network
        self.net_bases, self.scales, self.num_dim, self.dimidx2dimstr, aes_num = {}, [], 0, [], 0
        for k, v in opt['networks'].items():
            logger.info(f'Dim: {k}')
            net_base = build_network(v['network'])
            net_base = self.model_to_device(net_base)

            self.net_bases[k] = net_base
            if isinstance(v['scale'], list):
                scale_ = v['scale']
                scale1 = torch.tensor(scale_[0], dtype=torch.float32).to(self.device)
                scale2 = torch.tensor(scale_[1], dtype=torch.float32).to(self.device)
                self.scales.append([scale1, scale2])
            else:
                scale = torch.tensor(v['scale'], dtype=torch.float32).to(self.device)
                self.scales.append(scale)

            num_class = v['network']['num_class']

            for dim_idx in range(num_class):
                self.dimidx2dimstr.append(k + '_' + str(dim_idx + 1))

            self.num_dim += num_class
            if 'aes' in k:
                aes_num += num_class
        
        opt['network']['num_dim'] = self.num_dim
        opt['network']['aes_num'] = aes_num

        self.net_curr = build_network(opt['network'])
        self.net_curr = self.model_to_device(self.net_curr)
        self.print_network(self.net_curr)

        self.dist_key = {'SPAQ': 'dist', 'KADID10k': 'dist', 'PIPAL': 'dist', 'others': 'img'}
        self.gt_mos_key = {'SPAQ': 'label', 'KADID10k': 'label', 'PIPAL': 'label', 'others': 'mos_label'}
        self.dist_path_key = {'SPAQ': 'dist_path', 'KADID10k': 'dist_path', 'PIPAL': 'dist_path', 'others': 'img_path'}

        if self.is_train:
            self.init_training_settings()

        if opt['path'].get('pretrain_network', None) is not None:
            self.load_network(self.net_curr, opt['path'].get('pretrain_network', None), False, param_key='params')
            # for name, p in self.net_curr.named_parameters():
            #     if 'net' in name:
            #         p.requires_grad = False
            #         print('freeze', name)

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

        self.l_total = 0
        self.loss_dict = OrderedDict()

    def setup_optimizers(self):
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

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self.get_optimizer(optim_type, optim_params, **train_opt['optim'])
        self.optimizers.append(self.optimizer)
    
    def net_forward(self, net, aes=False):
        if aes:
            return net(self.dist_aes)
        return net(self.dist)
    
    def custom_feed_data(self, data, dataset_name):
        if dataset_name in self.dist_key.keys():
            dist_key = self.dist_key[dataset_name]
            gt_mos_key = self.gt_mos_key[dataset_name]
        else:
            dist_key = self.dist_key['others']
            gt_mos_key = self.gt_mos_key['others']
        self.dist = data[dist_key].to(self.device)
        if dist_key + '_aes' not in data.keys():
            self.dist_aes = data[dist_key].to(self.device)
        else:
            self.dist_aes = data[dist_key + '_aes'].to(self.device)
        # dist_paths = data['img_path']  # key: dist_path or img_path

        scaled_global_scores, scaled_local_scores, scaled_final_tokens = [], [], []
        with torch.no_grad():
            for idx, (dim, net_base) in enumerate(self.net_bases.items()):
                global_score_list, local_score_list, final_token_list = self.net_forward(net_base, aes='aes' in dim)
                # [B,1]  [B,d,h,w]
                scale = self.scales[idx]
                
                if isinstance(scale, list):
                    scale1, scale2 = scale[0], scale[1]
                    for gs in global_score_list:
                        scaled_global_score = (gs - scale1) / (scale2 - scale1)
                        scaled_global_scores.append(scaled_global_score)
                else:
                    for gs in global_score_list:
                        scaled_global_score = gs / scale
                        scaled_global_scores.append(scaled_global_score)
                for ls in local_score_list:
                    scaled_local_scores.append(ls)
                for ls in final_token_list:
                    scaled_final_tokens.append(ls)
        
        scaled_global_scores = torch.cat(scaled_global_scores, dim=1)  # B,(N1+N2+...)
        scaled_scores = [scaled_global_scores, scaled_local_scores, scaled_final_tokens]

        dataset_idx = self.datasets_idx[dataset_name] if self.dataset_specific_weighted else 0
        output_score = self.net_curr(self.dist, scaled_scores, weight_ratio=None, weights_idx=dataset_idx)  # TODO
        output_score1, output_score2 = score_dim_process(output_score)

        if dataset_name in ['koniq10k', 'KADID10k', 'SPAQ', 'PIPAL', 'FLIVE', 'livechallenge', 'BID']:
            gt_mos = data[gt_mos_key].to(self.device).type(output_score.dtype)
            gt_mos1, gt_mos2 = score_dim_process(gt_mos)

        for loss in self.dataset2loss[dataset_name]:
            if loss['type'] == 'FidelityLoss':
                l = loss['cri'](output_score1, gt_mos1)
            elif loss['type'] == 'NiNLoss':
                l = loss['cri'](output_score2, gt_mos2)
            elif loss['type'] == 'MSELoss':
                l = loss['cri'](output_score2, gt_mos2)
            elif loss['type'] == 'PLCCLoss':
                l = loss['cri'](output_score2, gt_mos2)
            self.l_total += l
            if loss['type'] in self.loss_dict.keys():
                self.loss_dict[loss['type']] += l
            else:
                self.loss_dict[loss['type']] = l

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        self.l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(self.loss_dict)

        self.l_total = 0
        self.loss_dict = OrderedDict()

    def test(self, dataset_idx=0):
        self.net_curr.eval()

        with torch.no_grad():
            scaled_global_scores, scaled_local_scores, scaled_final_tokens, global_scores = [], [], [], []
            for idx, (dim, net_base) in enumerate(self.net_bases.items()):
                global_score_list, local_score_list, final_token_list = self.net_forward(net_base, aes='aes' in dim)
                # B,1  B,h,w  B,h,w(if existed)
                scale = self.scales[idx]

                if isinstance(scale, list):
                    scale1, scale2 = scale[0], scale[1]
                    for gs in global_score_list:
                        scaled_global_score = (gs - scale1) / (scale2 - scale1)
                        scaled_global_scores.append(scaled_global_score)
                        global_scores.append(gs)
                else:
                    for gs in global_score_list:
                        scaled_global_score = gs / scale
                        scaled_global_scores.append(scaled_global_score)
                        global_scores.append(gs)
                for ls in local_score_list:
                    scaled_local_scores.append(ls)
                for ls in final_token_list:
                    scaled_final_tokens.append(ls)

        scaled_global_scores = torch.cat(scaled_global_scores, dim=1)  # B,(N1+N2+...)
        global_scores = torch.cat(global_scores, dim=1)  # B,(N1+N2+...)
        scaled_scores = [scaled_global_scores, scaled_local_scores, scaled_final_tokens]

        self.output_score = self.net_curr(self.dist, scaled_scores, weight_ratio=None, weights_idx=dataset_idx)  # TODO
        self.multidim_score = global_scores

        self.net_curr.train()

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

        pred_score = []
        pred_score_multidim = []
        gt_mos = []

        if dataset_name in self.dist_key.keys():
            dist_key = self.dist_key[dataset_name]
            gt_mos_key = self.gt_mos_key[dataset_name]
            dist_path_key = self.dist_path_key[dataset_name]
        else:
            dist_key = self.dist_key['others']
            gt_mos_key = self.gt_mos_key['others']
            dist_path_key = self.dist_path_key['others']

        dataset_idx = self.datasets_idx[dataset_name] if self.dataset_specific_weighted else 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.basename(val_data[dist_path_key][0])
            self.dist = val_data[dist_key].to(self.device)
            if dist_key + '_aes' not in val_data.keys():
                self.dist_aes = val_data[dist_key].to(self.device)
            else:
                self.dist_aes = val_data[dist_key + '_aes'].to(self.device)

            self.gt_mos = val_data[gt_mos_key].to(self.device)
            self.test(dataset_idx=dataset_idx)
            pred_score.append(self.output_score)
            # print(val_data[dist_path_key][0], self.output_score)
            pred_score_multidim.append(self.multidim_score)
            gt_mos.append(self.gt_mos)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name:>20}')

        if use_pbar:
            pbar.close()

        pred_score = torch.cat(pred_score, dim=0).squeeze(1).cpu().numpy()
        pred_score_multidim = list(torch.cat(pred_score_multidim, dim=0).chunk(self.num_dim, dim=1))
        pred_score_multidim = [p.squeeze(1).cpu().numpy() for p in pred_score_multidim]
        gt_mos = torch.cat(gt_mos, dim=0)
        gt_mos = gt_mos.squeeze(1).cpu().numpy() if gt_mos.ndim == 2 else gt_mos.cpu().numpy()

        if with_metrics:

            save_dir = os.path.join(self.opt['path']['experiments_root'], 'figures', f'iter{current_iter}')
            os.makedirs(save_dir, exist_ok=True)

            # calculate all metrics
            for i in range(self.num_dim):
                plt.cla()
                plt.scatter(pred_score_multidim[i], gt_mos)
                # calculate all metrics
                num = 1
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] = calculate_metric([pred_score_multidim[i], gt_mos], opt_)
                    plt.text(0.05, 1.0 - 0.05 * num, f'{name}: {self.metric_results[name]:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                    num += 1

                save_filename = f'{dataset_name}_{self.dimidx2dimstr[i]}.png'
                save_path = os.path.join(save_dir, save_filename)
                plt.savefig(save_path)

            plt.cla()
            plt.scatter(pred_score, gt_mos)
            num = 1
            for name, opt_ in self.opt['val']['metrics'].items():
                self.metric_results[name] = calculate_metric([pred_score, gt_mos], opt_)
                plt.text(0.05, 1.0 - 0.05 * num, f'{name}: {self.metric_results[name]:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                num += 1

            save_filename = f'{dataset_name}.png'
            save_path = os.path.join(save_dir, save_filename)
            plt.savefig(save_path)


            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_curr, self.net_best)
                    self.save_network(self.net_best, 'net_best')
                
                if self.update_until_better is not False:  # TODO
                    logger = get_root_logger()
                    if to_update or current_iter < self.update_until_better:
                        logger.info('Update best checkpoint.')
                    else:
                        self.copy_model(self.net_best, self.net_curr)
                        logger.info('Resume from last best checkpoint.')

            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_curr, self.net_best)
                    self.save_network(self.net_best, 'net_best')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
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

    def save(self, epoch, current_iter, save_net_label='net'):
        self.save_network(self.net_curr, save_net_label, current_iter)
        self.save_training_state(epoch, current_iter)


def score_dim_process(score):
    if score.ndim == 1:
        return score, score[:, None]
    elif score.ndim == 2:
        return score[:, 0], score