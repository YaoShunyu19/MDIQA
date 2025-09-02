import pandas as pd


csv_files = ['/root/ysy/BasicSR/results/ESRGAN_x4_f64b23_DIV2K_400k_B16G1_1.0IQALoss_load_pretrain_ESRGAN_iter5w/Set14.csv',
             '/root/ysy/BasicSR/results/ESRGAN_x4_f64b23_DIV2K_400k_B16G1_iter40w/Set14.csv']


weights = {
    'psnr': 0,
    'ssim': 0,
    'lpips': 0,
    'clipiqa': 1,
    'musiq': 1,
    'maniqa': 1
}

higher_is_better = {
    'psnr': True,
    'ssim': True,
    'lpips': False,
    'clipiqa': True,
    'musiq': True,
    'maniqa': True
}


def preprocess_results(df):
    for metric in df.columns[1:]:
        if not higher_is_better[metric]:
            df[metric] = -df[metric]

    return df


def load_and_process_algorithms():
    processed_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        df_processed = preprocess_results(df)
        processed_dfs.append(df_processed)

    return processed_dfs


def select_best_samples_by_avg_diff(processed_dfs):
    diff_samples = []

    first_algorithm_df = processed_dfs[0]

    for index in range(len(first_algorithm_df)):  # for each image
        score_diffs = []

        # 计算第一个算法和其他算法的加权得分优势
        for i in range(1, len(processed_dfs)):  # for each algorithm
            score_diff = []
            for metric_name in weights.keys():  # for each metric
                ratio =first_algorithm_df.iloc[index][metric_name] / processed_dfs[i].iloc[index][metric_name]
                score_diff.append(ratio * weights[metric_name])

            score_diff = sum(score_diff) / sum(weights.values())
            score_diffs.append(score_diff)

        avg_diff = sum(score_diffs) / len(score_diffs)
        diff_samples.append((first_algorithm_df.iloc[index], avg_diff))

    diff_samples_sorted = sorted(diff_samples, key=lambda x: x[1], reverse=True)

    best_samples_sorted = pd.DataFrame([x[0] for x in diff_samples_sorted])

    return best_samples_sorted


def main():

    processed_dfs = load_and_process_algorithms()

    best_samples_sorted = select_best_samples_by_avg_diff(processed_dfs)

    print(best_samples_sorted[['img_name']])


main()
