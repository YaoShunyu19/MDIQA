import argparse
import cv2
import glob
import numpy as np
import os
import torch
import pandas as pd
import pyiqa
from PIL import Image
import matplotlib.pyplot as plt
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.rrdbnet_condition_arch import RRDBNet_Condition
from basicsr.losses.iqa_model_loss import MultiDimSharedIQAModel

metrics = {
    'psnr': 'FR',
    'ssim': 'FR',
    'lpips': 'FR',
    'clipiqa': 'NR',
    'musiq': 'NR',
    'maniqa': 'NR',
    'niqe': 'NR'
}
multi_dims = ['brightness', 'colorfulness',
              'contrast', 'noise',
              'sharpness', 'color',
              'composition', 'content',
              'light', 'overall']

def main():

    names = [
        # 'finetune_RealESRGANx4plus_lr3e-5_B16G1',
        # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0IQALoss',
        # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_weightbranch_nograd'

        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss',

        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_1.5sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_2.0sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_2.5sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_3.0sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_4.0sharp',

        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_1.1noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_1.3noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_1.5noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_1.7noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_0.4mbv3cliplocalNRIQALoss_2.0noise',


        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss',

        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_1.5sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_2.0sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_2.5sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_3.0sharp',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_4.0sharp',

        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_1.1noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_1.3noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_1.5noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_1.7noise',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_70%_2.0mbv3cliplocalFRIQALoss_2.0noise',

        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0topiq_nr',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0topiq_fr',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_0.8topiq_nr_fr',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_0.6clipiqa',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_0.6clipiqa+',

        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0NRIQALoss_1.3colorfulness',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0NRIQALoss_1.6colorfulness',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0NRIQALoss_1.9colorfulness',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0NRIQALoss_2.2colorfulness',
        # 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0NRIQALoss_2.5colorfulness',

        'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0NRIQALoss_sharpness1-4',
    ]
    # ckpts = range(5000, 120000, 5000)

    # names = [
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0IQALoss',

    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_1.5sharp',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_2sharp',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_2.5sharp',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_3sharp',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_4sharp',

    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_1.1noise',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_1.3noise',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_1.5noise',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_1.7noise',
    #     # 'experiments_NR/finetune_RealESRGANx4plus_lr3e-5_B12G1_2noise',


    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_weightbranch_nograd',

    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_1.5sharp_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_2.0sharp_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_2.5sharp_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_3.0sharp_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_4.0sharp_weightbranch_nograd',

    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_1.1noise_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_1.3noise_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_1.5noise_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_1.7noise_weightbranch_nograd',
    #     # 'experiments_FR/finetune_RealESRGANx4plus_lr3e-5_B12G1_5.0weightedFRIQALoss_2.0noise_weightbranch_nograd',
    # ]

    ckpts = [25000]

    ckpts = [str(ckpt) for ckpt in ckpts]
    params_keys = 'params'
    x = ''  # _50
    inputs = [
        f'experiments/generate_RealESRGAN_mixed_deg_images{x}/visualization/lq',
        # f'/root/ysy/BasicSR/experiments/generate_RealESRGAN_only_blur_images{x}/visualization/lq',
        # f'/root/ysy/BasicSR/experiments/generate_RealESRGAN_only_noise_images{x}/visualization/lq',
    ]

    for name in names:
        for input in inputs:
            for ckpt in ckpts:
                model_path = os.path.join('experiments', name, 'models', f'net_g_{ckpt}.pth')

                if params_keys == 'params_ema':
                    name_ = name + '_ema'
                else:
                    name_ = name
                output_path = os.path.join('results', name_, ckpt)

                input_name = input.split('/')[-3]
                if not os.path.exists(os.path.join(output_path, 'visualization', input_name)):
                    inference(model_path, output_path, input, input_name, params_keys)
                calculate_metrics(output_path, input_name)

            output_path = os.path.join('results', name_)
            csv_names = [input_name + '.csv', input_name + '_multidim.csv']
            plot(output_path, ckpts, csv_names)


@torch.no_grad()
def inference(model_path, output_path, input, input_name, params_keys='params'):
    output_dir = os.path.join(output_path, 'visualization', input_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model = RRDBNet_Condition(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(torch.load(model_path)[params_keys], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                # output = model(img)
                weight_ratio = torch.ones(9)
                weight_ratio[4] = 4.0
                output = model(img, weight_ratio)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'{imgname}.png'), output)


@torch.no_grad()
def calculate_metrics(output_path, input_name):
    output_dir = os.path.join(output_path, 'visualization', input_name)
    img_list = sorted(glob.glob(os.path.join(output_dir, '*')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iqa_metrics = [pyiqa.create_metric(metric_name, device=device) for metric_name in metrics.keys()]
    multidim_iqa_model = MultiDimSharedIQAModel(
        multidim_fusion_net_pretrained_model_path='../IQA-PyTorch/local_training_20240727/experiments/benchmark_koniq10k/shared_5t_4a_clipfeat_localv4_mbv3weight_koniq_official/models/net_best.pth',
        weight_model_name='mobilenetv3_large_100.ra_in1k'
    ).to(device)

    iqa_all = {metric_name: [] for metric_name in metrics.keys()}
    multidim_iqa_all = {dim_name: [] for dim_name in multi_dims}
    save_img_names = []

    for i, img_path in enumerate(img_list):
        basename = os.path.basename(img_path)
        save_img_names.append(basename)

        # calculate iqa metric
        for idx, metric_name in enumerate(metrics.keys()):
            fr_or_nr = metrics[metric_name]
            metric = iqa_metrics[idx]

            if fr_or_nr.lower() == 'nr':
                iqa_val = metric(img_path).item()
            elif fr_or_nr.lower() == 'fr':
                gt_path = os.path.join('experiments', img_path.split('/')[4], 'visualization', 'gt', basename)  # TODO if in experiments_N/FR: 5 else: 4
                iqa_val = metric(img_path, gt_path).item()
            else:
                print('wrong.')
            print(f'{i+1:3d}: {basename:25}. \t{metric_name}: {iqa_val:.6f}.')
            iqa_all[metric_name].append(iqa_val)

        img_pil = Image.open(img_path).convert('RGB')
        img = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.
        multidim_scores, overall_score = multidim_iqa_model(img)
        multidim_scores = torch.cat([multidim_scores, overall_score], dim=1)
        for idx, dim_name in enumerate(multi_dims):
            score = multidim_scores[0, idx].item()
            print(f'{i+1:3d}: {basename:25}. \t{dim_name}: {score:.6f}.')
            multidim_iqa_all[dim_name].append(score)

    for metric_name in metrics.keys():
        avg_metric_value =sum(iqa_all[metric_name]) / len(iqa_all[metric_name])
        iqa_all[metric_name].append(avg_metric_value)
        print(f'Average: {metric_name}: {avg_metric_value:.6f}')

    for dim_name in multi_dims:
        avg_metric_value =sum(multidim_iqa_all[dim_name]) / len(multidim_iqa_all[dim_name])
        multidim_iqa_all[dim_name].append(avg_metric_value)
        print(f'Average: {dim_name}: {avg_metric_value:.6f}')


    save_img_names.append('average')

    data = {'img_name': save_img_names}
    for metric_name in iqa_all.keys():
        data[metric_name] = iqa_all[metric_name]
    df = pd.DataFrame(data)
    df.to_csv(output_dir.replace('/visualization/', '/') + '.csv', index=False)

    data = {'img_name': save_img_names}
    for dim_name in multidim_iqa_all.keys():
        data[dim_name] = multidim_iqa_all[dim_name]
    df = pd.DataFrame(data)
    df.to_csv(output_dir.replace('/visualization/', '/') + '_multidim.csv', index=False)

    # if 'fid' in metrics.keys():
    # fid_metric = pyiqa.create_metric('fid')
    # folder_gt = os.path.join('experiments', output_dir.split('/')[4], 'visualization', 'gt/')
    # fid_score = fid_metric(output_dir, folder_gt)
    # print(f'Average: FID: {fid_score:.6f}')
    # data = {'FID': [fid_score]}
    # df = pd.DataFrame(data)
    # df.to_csv(output_dir.replace('/visualization/', '/') + '_fid.csv', index=False)


def plot(output_path, sub_folders, csv_names):

    for csv_name in csv_names:
        save_dir = os.path.join(output_path, 'plot', csv_name.split('.')[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        metrics_values = {}

        for sub_folder in sub_folders:
            csv_file_path = os.path.join(output_path, sub_folder, csv_name)

            data = pd.read_csv(csv_file_path)
            metric_names = list(data.columns)[1:]

            for metric_name in metric_names:
                if metric_name in metrics_values.keys():
                    metrics_values[metric_name].append(data[metric_name].iloc[-1])
                else:
                    metrics_values[metric_name] = [data[metric_name].iloc[-1]]

        for metric_name in metrics_values.keys():
            plot_name = metric_name + '.png'
            plt.figure(figsize=(10, 6))
            plt.plot(sub_folders, metrics_values[metric_name], label=metric_name, marker='o')
            plt.xlabel('Iterations')
            plt.ylabel('Metric Value')
            plt.title('Performance Metrics across Iterations')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, plot_name), dpi=300)
            plt.close()


if __name__ == '__main__':
    main()
