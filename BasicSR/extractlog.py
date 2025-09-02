import re

# 打开log文件
log_file_path = "realesrgan_noise0.5_sharp0.5_5w_realsr.txt"
with open(log_file_path, "r", encoding="utf-8") as file:
    log_content = file.read()

# 匹配每张图像的9维评分和综合评分
# dimension_scores_pattern = r"tensor\(\[\[([\d\.,\s]+)\]\],\s+device='cuda:0',\s+grad_fn=<CatBackward0>\)"
# overall_score_pattern = r"tensor\(\[\[([\d\.]+)\]\],\s+device='cuda:0',\s+grad_fn=<AddmmBackward0>\)"

dimension_scores_pattern = r"tensor\(\[\[([\d\.,\s]+,[\d\.,\s]+)\]\],\s+device='cuda:0'\)"
overall_score_pattern = r"tensor\(\[\[([\d\.]+)\]\],\s+device='cuda:0'\)"


# 提取9维评分和综合评分
dimension_scores_matches = re.findall(dimension_scores_pattern, log_content)
overall_score_matches = re.findall(overall_score_pattern, log_content)
# 转换为浮点数
dimension_scores_list = [list(map(float, match.split(", "))) for match in dimension_scores_matches]
overall_scores = list(map(float, overall_score_matches))

# 计算每个维度的均分
num_dimensions = 9
average_dimension_scores = [sum(scores[i] for scores in dimension_scores_list) / len(dimension_scores_list) for i in range(num_dimensions)]

# 计算综合评分的均分
average_overall_score = sum(overall_scores) / len(overall_scores)

# 打印结果
print("每个维度的均分：")
for i, avg_score in enumerate(average_dimension_scores, 1):
    print(f"维度 {i}: {avg_score:.4f}")

print(f"\n综合评分的均分：{average_overall_score:.4f}")
