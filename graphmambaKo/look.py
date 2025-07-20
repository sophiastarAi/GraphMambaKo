import csv

# # 定义CSV文件路径,查看csv文件维度
# file_path = 'data/2018_2019_hourly_test_x.csv'
# row_count = 0
# col_count = None
#
# # 打开CSV文件
# with open(file_path, 'r', encoding='utf-8') as file:
#     # 创建CSV读取器对象
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         # 统计行数
#         row_count += 1
#         if col_count is None:
#             # 获取列数，仅在第一行时获取
#             col_count = len(row)
#
# print(f"行数: {row_count}")
# print(f"列数: {col_count}")

#拼接文件
import pandas as pd


file1_path = 'data/2018_2019_hourly.csv'
file2_path = 'data/2018_2019_hourly_val_x.csv'
file3_path = 'data/2018_2019_hourly_test_x.csv'


# df1 = pd.read_csv(file1_path)
# df2 = pd.read_csv(file2_path)
# df3 = pd.read_csv(file3_path)

df1 = pd.read_csv(file1_path, sep=',', encoding='utf-8', header=None )
df2 = pd.read_csv(file2_path, sep=',', encoding='utf-8', header=None)
df3 = pd.read_csv(file3_path, sep=',', encoding='utf-8', header=None)


print(f"df1 shape: {df1.shape}")
print(f"df2 shape: {df2.shape}")
print(f"df3 shape: {df3.shape}")


combined_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)


print(f"Combined DataFrame shape: {combined_df.shape}")


output_file = 'data/combined_1819_hourly.csv'
combined_df.to_csv(output_file, index=False, header=False)

print(f"Combined data saved to {output_file}")