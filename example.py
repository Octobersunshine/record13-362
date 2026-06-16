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
print("各区间样本数（等频应接近 100/5=20）：")
print(discretizer.get_bin_counts(result2, '收入'))
print()
print("各区间占比：")
print(discretizer.get_bin_counts(result2, '收入', normalize=True).round(4) * 100)
print()

print("=" * 60)
print("示例2-增强：等宽 vs 等频 样本分布对比")
print("=" * 60)
disc_width = DataDiscretizer()
res_width = disc_width.fit_transform(
    data, columns=['收入'], method='equal_width', n_bins=5,
    labels={'收入': ['组1', '组2', '组3', '组4', '组5']}
)
disc_freq = DataDiscretizer()
res_freq = disc_freq.fit_transform(
    data, columns=['收入'], method='equal_freq', n_bins=5,
    labels={'收入': ['组1', '组2', '组3', '组4', '组5']}
)
compare = pd.DataFrame({
    '等宽_样本数': disc_width.get_bin_counts(res_width, '收入'),
    '等宽_占比%': (disc_width.get_bin_counts(res_width, '收入', normalize=True) * 100).round(2),
    '等频_样本数': disc_freq.get_bin_counts(res_freq, '收入'),
    '等频_占比%': (disc_freq.get_bin_counts(res_freq, '收入', normalize=True) * 100).round(2),
})
print(compare)
print()
print("等宽分箱详情：")
print(disc_width.get_bin_info('收入'))
print()
print("等频分箱详情（describe_equal_freq 详细报告）：")
print(disc_freq.describe_equal_freq(res_freq, '收入'))
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
print()

print("=" * 60)
print("示例8：左闭右开区间 (right=False) - 年龄按 [低, 中, 高) 归属")
print("=" * 60)
discretizer8 = DataDiscretizer()
result8 = discretizer8.fit_transform(
    data,
    columns=['年龄'],
    method='equal_width',
    n_bins=3,
    right=False,
    include_lowest=True,
    labels={'年龄': ['青年', '中年', '老年']}
)
print("右闭左开 vs 左闭右开 对比：")
print(result8[['年龄', '年龄_discretized']].head(10))
print()
print("分箱详情（right=False，左闭右开）：")
print(discretizer8.get_bin_info('年龄'))
print()

print("=" * 60)
print("示例9：边界值归属验证 - 精确测试区间开闭")
print("=" * 60)
test_data = pd.DataFrame({'score': [0, 60, 60.0, 80, 80.0, 100]})
print(f"测试数据: {test_data['score'].tolist()}")
print()

print("--- right=True（默认：左开右闭 (a, b]）---")
disc_right = DataDiscretizer()
res_right = disc_right.fit_transform(
    test_data,
    columns=['score'],
    method='custom',
    custom_bins={'score': [0, 60, 80, 100]},
    labels={'score': ['不及格', '及格', '优秀']},
    right=True,
    include_lowest=True
)
print(res_right[['score', 'score_discretized']])
print(disc_right.get_bin_info('score'))
print()

print("--- right=False（左闭右开 [a, b)）---")
disc_left = DataDiscretizer()
res_left = disc_left.fit_transform(
    test_data,
    columns=['score'],
    method='custom',
    custom_bins={'score': [0, 60, 80, 100]},
    labels={'score': ['不及格', '及格', '优秀']},
    right=False,
    include_lowest=True
)
print(res_left[['score', 'score_discretized']])
print(disc_left.get_bin_info('score'))
print()

print("关键差异：")
print(f"  score=60 在 right=True  时归属: {res_right.loc[1, 'score_discretized']}（60 落入 (0,60] 右闭区间）")
print(f"  score=60 在 right=False 时归属: {res_left.loc[1, 'score_discretized']}（60 落入 [60,80) 左闭区间）")
print(f"  score=80 在 right=True  时归属: {res_right.loc[3, 'score_discretized']}（80 落入 (60,80] 右闭区间）")
print(f"  score=80 在 right=False 时归属: {res_left.loc[3, 'score_discretized']}（80 落入 [80,100) 左闭区间）")
print()

print("=" * 60)
print("示例10：等频离散化高级场景 - 大量重复值（订单金额）")
print("=" * 60)
np.random.seed(123)
repeated_data = pd.DataFrame({
    '订单金额': np.random.choice([9.9, 19.9, 29.9, 49.9, 99.9, 199.9, 299.9, 499.9, 999.9],
                                  size=500,
                                  p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02])
})
print(f"订单金额数据分布（前10行）：\n{repeated_data.head(10)}")
print(f"唯一值数量：{repeated_data['订单金额'].nunique()}，共{len(repeated_data)}条记录")
print(f"唯一值：{sorted(repeated_data['订单金额'].unique())}")
print()

disc_rep = DataDiscretizer()
res_rep = disc_rep.fit_transform(
    repeated_data,
    columns=['订单金额'],
    method='equal_freq',
    n_bins=4,
    labels={'订单金额': ['低价', '中低价', '中高价', '高价']}
)
print("等频离散化结果统计：")
print(disc_rep.get_bin_counts(res_rep, '订单金额'))
print()
print("等频分箱详情：")
print(disc_rep.describe_equal_freq(res_rep, '订单金额'))
print()

print("=" * 60)
print("示例11：等频离散化高级场景 - 偏态分布（用户资产/收入长尾）")
print("=" * 60)
np.random.seed(777)
skewed_data = pd.DataFrame({
    '用户资产': np.random.exponential(scale=50000, size=1000).round(2)
})
print(f"偏态数据统计：\n{skewed_data['用户资产'].describe().round(2)}")
print(f"数据偏度: {skewed_data['用户资产'].skew().round(2)}（正偏态，长尾）")
print()

disc_skew = DataDiscretizer()
res_skew = disc_skew.fit_transform(
    skewed_data,
    columns=['用户资产'],
    method='equal_freq',
    n_bins=5,
    labels={'用户资产': ['大众', '小康', '富裕', '高净值', '超高净值']}
)
print("等频离散化结果统计（每档应接近 200 条）：")
print(disc_skew.get_bin_counts(res_skew, '用户资产'))
print()
print("等频分箱详情（偏态数据的区间宽度差异显著）：")
print(disc_skew.describe_equal_freq(res_skew, '用户资产'))
print()

print("=" * 60)
print("示例12：等频插值方法对比 (linear vs nearest vs lower vs higher)")
print("=" * 60)
small_data = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
})
print(f"测试数据: {small_data['value'].tolist()}")
print()

for interp in ['linear', 'lower', 'higher', 'midpoint', 'nearest']:
    d = DataDiscretizer()
    r = d.fit_transform(
        small_data,
        columns=['value'],
        method='equal_freq',
        n_bins=3,
        interpolation=interp,
        labels={'value': ['A', 'B', 'C']}
    )
    print(f"--- interpolation='{interp}' ---")
    print(d.get_bin_info('value')[['label', 'interval', 'bin_start', 'bin_end']])
    counts = d.get_bin_counts(r, 'value')
    print(f"样本分布: {dict(counts)}")
    print()

print("=" * 60)
print("示例13：get_bin_counts 支持多列批量统计")
print("=" * 60)
disc_multi = DataDiscretizer()
res_multi = disc_multi.fit_transform(
    data,
    columns=['年龄', '收入', '消费金额'],
    method='equal_freq',
    n_bins=3
)
multi_counts = disc_multi.get_bin_counts(res_multi)
for col, counts in multi_counts.items():
    print(f"{col}: {dict(counts)}")
