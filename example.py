import pandas as pd
import numpy as np
from discretization import DataDiscretizer

np.random.seed(42)

data = pd.DataFrame({
    '年龄': np.random.randint(18, 70, 100),
    '收入': np.random.randint(30000, 150000, 100),
    '消费金额': np.random.uniform(100, 5000, 100).round(2),
    '购物频率': np.random.randint(1, 30, 100)
})

print("=" * 60)
print("原始数据前10行：")
print("=" * 60)
print(data.head(10))
print()

discretizer = DataDiscretizer()

print("=" * 60)
print("示例1：等宽分箱 - 按自定义标签（低/中/高）转换年龄")
print("=" * 60)
result1 = discretizer.fit_transform(
    data,
    columns=['年龄'],
    method='equal_width',
    n_bins=3,
    labels={'年龄': ['青年', '中年', '老年']}
)
print(result1[['年龄', '年龄_discretized']].head(10))
print()
print("分箱详情：")
print(discretizer.get_bin_info('年龄'))
print()

print("=" * 60)
print("示例2：等频分箱 - 收入按等频分为5档")
print("=" * 60)
result2 = discretizer.fit_transform(
    data,
    columns=['收入'],
    method='equal_freq',
    n_bins=5,
    labels={'收入': ['很低', '低', '中', '高', '很高']}
)
print(result2[['收入', '收入_discretized']].head(10))
print()
print("分箱详情：")
print(discretizer.get_bin_info('收入'))
print()

print("=" * 60)
print("示例3：自定义区间分箱 - 消费金额按业务规则划分")
print("=" * 60)
custom_bins = {
    '消费金额': [0, 500, 1500, 3000, 5000]
}
custom_labels = {
    '消费金额': ['小额', '中额', '大额', '超大额']
}
result3 = discretizer.fit_transform(
    data,
    columns=['消费金额'],
    method='custom',
    custom_bins=custom_bins,
    labels=custom_labels
)
print(result3[['消费金额', '消费金额_discretized']].head(10))
print()
print("分箱详情：")
print(discretizer.get_bin_info('消费金额'))
print()

print("=" * 60)
print("示例4：K-Means聚类分箱 - 购物频率自动分组")
print("=" * 60)
result4 = discretizer.fit_transform(
    data,
    columns=['购物频率'],
    method='kmeans',
    n_bins=3,
    labels={'购物频率': ['低频', '中频', '高频']}
)
print(result4[['购物频率', '购物频率_discretized']].head(10))
print()
print("分箱详情：")
print(discretizer.get_bin_info('购物频率'))
print()

print("=" * 60)
print("示例5：一次性处理多列，使用默认标签")
print("=" * 60)
result5 = discretizer.fit_transform(
    data,
    columns=['年龄', '收入', '消费金额'],
    method='equal_width',
    n_bins=3
)
print(result5[['年龄', '年龄_discretized', '收入', '收入_discretized', '消费金额', '消费金额_discretized']].head(10))
print()
print("所有分箱信息：")
bin_info = discretizer.get_bin_info()
for col, info in bin_info.items():
    print(f"\n--- {col} ---")
    print(info)
print()

print("=" * 60)
print("示例6：使用fit-then-transform模式（适用于训练集/测试集场景）")
print("=" * 60)
train_data = data.iloc[:70].copy()
test_data = data.iloc[70:].copy()

discretizer2 = DataDiscretizer()
discretizer2.fit(
    train_data,
    columns=['年龄', '收入'],
    method='equal_width',
    n_bins=3,
    labels={'年龄': ['青年', '中年', '老年'], '收入': ['低', '中', '高']}
)

train_transformed = discretizer2.transform(train_data)
test_transformed = discretizer2.transform(test_data)

print("训练集转换结果前5行：")
print(train_transformed[['年龄', '年龄_discretized', '收入', '收入_discretized']].head())
print()
print("测试集转换结果前5行：")
print(test_transformed[['年龄', '年龄_discretized', '收入', '收入_discretized']].head())
print()

print("=" * 60)
print("示例7：统计各分类的样本数量")
print("=" * 60)
final_result = discretizer.fit_transform(
    data,
    columns=['年龄'],
    method='equal_width',
    n_bins=3,
    labels={'年龄': ['青年(18-35)', '中年(36-52)', '老年(53-70)']}
)
print("年龄分布统计：")
print(final_result['年龄_discretized'].value_counts().sort_index())
print()
print("年龄分布百分比：")
print((final_result['年龄_discretized'].value_counts(normalize=True).sort_index() * 100).round(2).astype(str) + '%')
