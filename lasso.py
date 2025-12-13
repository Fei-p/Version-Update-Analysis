from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def analyze_drivers(df, country_name, features, target='ad_revenue'):
    # 1. 筛选特定国家的数据
    data = df[df['country'] == country_name].copy()
    
    # 去除空值
    data = data.dropna(subset=features + [target])
    
    if len(data) < 50:
        print(f"{country_name} 数据太少，跳过")
        return None

    X = data[features]
    y = data[target]
    
    # 2. 标准化 (Lasso 必须做！)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 运行 Lasso
    model = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_scaled, y)
    
    # 4. 提取系数（完整 & 非零）
    coefs = pd.Series(model.coef_, index=features)

    # ============== 新增输出内容 ==============
    print("\n" + "="*60)
    print(f"国家：{country_name}")
    print(f"最佳 alpha: {model.alpha_}")
    print("====================================================")

    # （A）全部系数，按绝对值降序排序（便于理解重要性）
    print("\n【全部系数（按绝对值从大到小排序）】")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index))

    # （B）非零系数
    important_coefs = coefs[coefs != 0].sort_values()
    print(f"\n【非零系数（Lasso 选中的变量，共 {len(important_coefs)} 个）】")
    print(important_coefs)
    # ==========================================

    return important_coefs


# === 设置你的特征列表 ===
df = pd.read_csv(r"C:\Users\HP\Downloads\final-version-data-augmented.csv")
# 记得把那些你新造的指标都放进去
# 1. 首先处理分类变量 (One-Hot)
# 注意：这一步要在筛选 columns 之前做
df_model = pd.get_dummies(df, columns=['initial_version', 'is_weekday'], drop_first=True)

# 2. 定义特征列表 (X)
feature_list = [
    # --- 时间与粘性 ---
    'retention_day',              # 第几天留存
    'avg_duration_per_session',   # 每次看多久 (深度)
    'session_open',               # 一天看几次 (频次 - 这是一个关键指标！)
    
    # --- 页面行为 (细分) ---
    # 去掉 total_page，看具体哪个页面贡献大
    'page_home', 
    'page_search', 
    'page_forecast', 
    'page_radar',
    'avg_page_per_session',       # 浏览深度
    
    # --- 交互行为 ---
    'click_unlock',               # 解锁
    
    # --- 广告相关 (只放比率和密度，不放绝对值) ---
    'ad_density',                 # 广告密度 (Impression / Duration)
    'ad_click_through_rate',      # CTR (Click / Impression)
    'rewarded_ratio',             # 激励视频占比 (主动看广告的意愿)
    
    # --- 虚拟变量 (由 pd.get_dummies 生成的) ---
    # 比如你的版本号是 1.2.0, 1.2.9... 代码会自动生成 initial_version_1.2.9
    # 这里你需要把所有生成的列名加进来，或者用代码自动抓取
]

# 自动抓取所有 initial_version 开头的列
initial_ver_cols = [col for col in df_model.columns if 'initial_version_' in col]
is_weekday_cols = [col for col in df_model.columns if 'is_weekday_' in col] # 如果 is_weekday 已经是 0/1 则不需要这步

# 最终的 X 特征池
final_features = feature_list + initial_ver_cols

# === 分别运行 ===
print("正在分析 US (表现不佳)...")
us_drivers = analyze_drivers(df_model, 'USA', final_features)

print("正在分析 India (表现较好)...")
in_drivers = analyze_drivers(df_model, 'India', final_features)

# === 画图对比 ===
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

if us_drivers is not None:
    us_drivers.plot(kind='barh', ax=axes[0], color='salmon')
    axes[0].set_title('US: Drivers of Revenue (Why did it fail?)')
    axes[0].axvline(0, color='black', linestyle='--')

if in_drivers is not None:
    in_drivers.plot(kind='barh', ax=axes[1], color='lightgreen')
    axes[1].set_title('India: Drivers of Revenue (Why did it succeed?)')
    axes[1].axvline(0, color='black', linestyle='--')

plt.tight_layout()
plt.show()