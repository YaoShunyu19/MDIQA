import glob
import os.path as osp
import torch
import pyiqa
import pandas as pd
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

def read_image(path, centor_crop=0.5):
    img = Image.open(path).convert('RGB')
    img_tensor = transform(img)
    if centor_crop > 0:
        _, h, w = img_tensor.shape
        hh, ww = h // 2, w // 2
        img_tensor = img_tensor[:, hh - int(0.5 * centor_crop * h): hh + int(0.5 * centor_crop * h), ww - int(0.5 * centor_crop * w): ww + int(0.5 * centor_crop * w)]
    return img_tensor[None, ...]

def main():
    # Configurations
    # -------------------------------------------------------------------------
    # TODO: suffix/dataset_name
    # suffix = 'ESRGAN_PSNR_SRx4_DIV2K_iter100w'
    # suffix = 'ESRGAN_x4_f64b23_DIV2K_400k_B16G1_iter40w'
    # suffix = 'ESRGAN_woGANLoss_x4_f64b23_DIV2K_400k_B16G1_iter25w'
    # suffix = 'ESRGAN_x4_f64b23_DIV2K_400k_B16G1_1.0IQALoss_iter5w_ema'
    # suffix = 'ESRGAN_woGANLoss_x4_f64b23_DIV2K_400k_B16G1_0.05IQALoss_iter30w'
    # suffix = 'ESRGAN_x4_f64b23_DIV2K_400k_B16G1_1.0IQALoss_0.05-0.05-0.05-0.3-0.35-0.05-0.05-0.05-0.05_iter10w'

    # suffix = 'finetune_RealESRGANx4plus_lr3e-5_B16G1'
    # suffix = 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0IQALoss_iter9.5w'
    suffix = 'finetune_RealESRGANx4plus_lr3e-5_B12G1_1.0LightLoss_iter3w'

    datasets = [5,6]

    for d in datasets:
        calculate_metrics(suffix, d)

metrics = {
        # 'psnr': 'FR',
        # 'ssim': 'FR',
        # 'lpips': 'FR',
        'clipiqa': 'NR',
        'musiq': 'NR',
        'maniqa': 'NR'
}

@torch.no_grad()
def calculate_metrics(suffix, dataset):
    if dataset == 1:
        folder_gt = 'datasets/Set5/GTmod12'
        folder_restored = f'results/{suffix}/visualization/Set5'
        scale = ''
    elif dataset == 2:
        folder_gt = 'datasets/Set14/GTmod12'
        folder_restored = f'results/{suffix}/visualization/Set14'
        scale = ''
    elif dataset == 3:
        folder_gt = 'datasets/DIV2K/DIV2K_valid_HR'
        folder_restored = f'results/{suffix}/visualization/DIV2K100'
        scale = 'x4'
    elif dataset == 4:
        folder_restored = f'results/{suffix}/visualization/OutdoorSceneTest300'
        folder_gt = folder_restored
    elif dataset == 5:
        folder_restored = f'results/{suffix}/visualization/DPED_sample_patches_subset'
        folder_gt = folder_restored
    elif dataset == 6:
        folder_restored = f'results/{suffix}/visualization/RealSR_test_x4_subset'
        folder_gt = folder_restored
    # crop_border = 4
    # -------------------------------------------------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(pyiqa.list_models())

    iqa_metrics = [pyiqa.create_metric(metric_name, device=device) for metric_name in metrics.keys()]
    for metric in iqa_metrics:
        print(metric.lower_better)

    iqa_all = {metric_name: [] for metric_name in metrics.keys()}
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    save_img_names = []

    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt_path = img_path
        if dataset in [4, 5, 6]:
            img_restored_path = img_gt_path
        else:
            img_restored_path = osp.join(folder_restored, basename + scale + '_' + suffix + ext)
        save_img_names.append(osp.basename(img_path))

        # calculate iqa metric
        for idx, metric_name in enumerate(metrics.keys()):
            fr_or_nr = metrics[metric_name]
            metric = iqa_metrics[idx]
            try:
                if fr_or_nr.lower() == 'fr':
                    iqa_val = metric(img_restored_path, img_gt_path).item()
                elif fr_or_nr.lower() == 'nr':
                    iqa_val = metric(img_restored_path).item()
                else:
                    print('wrong.')
                print(f'{i+1:3d}: {basename:25}. \t{metric_name}: {iqa_val:.6f}.')
                iqa_all[metric_name].append(iqa_val)
            except RuntimeError:
                img_restored_tensor = read_image(img_restored_path, 0.5)
                img_gt_tensor = read_image(img_gt_path, 0.5)
                if fr_or_nr.lower() == 'fr':
                    iqa_val = metric(img_restored_tensor, img_gt_tensor).item()
                elif fr_or_nr.lower() == 'nr':
                    iqa_val = metric(img_restored_tensor).item()
                else:
                    print('wrong.')
                print(f'{i+1:3d}: {basename:25}. \t{metric_name}: {iqa_val:.6f}.')
                iqa_all[metric_name].append(iqa_val)

    for metric_name in metrics.keys():
        avg_metric_value =sum(iqa_all[metric_name]) / len(iqa_all[metric_name])
        iqa_all[metric_name].append(avg_metric_value)
        print(f'Average: {metric_name}: {avg_metric_value:.6f}')


    if 'fid' in metrics.keys():
        fid_metric = pyiqa.create_metric('fid')
        fid_score = fid_metric(folder_restored, folder_gt)
        print(f'Average: FID: {fid_score:.6f}')

    print(folder_restored)

    save_img_names.append('average')
    data = {'img_name': save_img_names}
    for metric_name in iqa_all.keys():
        data[metric_name] = iqa_all[metric_name]
    df = pd.DataFrame(data)
    df.to_csv(folder_restored.replace('/visualization/', '/') + '.csv', index=False)

if __name__ == '__main__':
    main()
