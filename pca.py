from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 准备数据：只选 US 和 India
df = pd.read_csv(r"C:\Users\HP\Downloads\final-version-data-augmented.csv")

df_compare = df[df['country'].isin(['USA', 'India'])].copy()

# 2. 选择纯行为特征 (不放 ID, Date, Revenue, Version)
# 我们想看的是“行为模式”的差异
pca_features = [
    'avg_duration_per_session', 
    'avg_page_per_session', 
    'session_open',          # 频次
    'ad_density',            # 广告密度
    'page_home',             # 首页依赖度 (Lasso 发现这个很重要)
    'page_radar',           # 雷达页依赖度
    'page_search',
    'page_forecast',
    'ad_click_through_rate',
    'total_click',         
    'rewarded_ratio'
]

# 去除空值
df_pca_clean = df_compare.dropna(subset=pca_features)

# 3. 标准化 (Standardization) - PCA 必须做
x = df_pca_clean[pca_features].values
x = StandardScaler().fit_transform(x)

# 4. 运行 PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

# 把国家标签拼回来
pca_df['Country'] = df_pca_clean['country'].values

# 5. 画图
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="PC1", y="PC2", 
    hue="Country", 
    data=pca_df, 
    alpha=0.4,       # 设置透明度，方便看重叠程度
    palette={'USA': 'salmon', 'India': 'lightgreen'}
)

plt.title('PCA: Behavioral Landscape (USA vs India)', fontsize=15)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# 6. 查看 PC1 和 PC2 到底代表什么 (Loadings)
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1', 'PC2'], 
    index=pca_features
)
print("--- PCA Loadings (成分含义) ---")
print(loadings)