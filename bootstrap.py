import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 定义 Bootstrap 函数
def run_bootstrap_analysis(df, metric_col, version_col='initial_version', 
                           control_ver='1.2.9', test_ver='1.3.3', n_iter=3000):
    """
    df: 你的数据表
    metric_col: 你要分析的指标，比如 'avg_duration_per_session'
    control_ver: 旧版本号 (s13)
    test_ver: 新版本号 (s14)
    """
    # 提取两组数据 (去除空值，防止报错)
    group_control = df[df[version_col] == control_ver][metric_col].dropna().values
    group_test = df[df[version_col] == test_ver][metric_col].dropna().values
    
    # 记录每次重采样的均值差
    diffs = []
    
    print(f"正在对 {metric_col} 进行 {n_iter} 次 Bootstrap 重采样...")
    
    for _ in range(n_iter):
        # 有放回抽样 (Resampling with replacement)
        sample_c = np.random.choice(group_control, size=len(group_control), replace=True)
        sample_t = np.random.choice(group_test, size=len(group_test), replace=True)
        
        # 计算：新版本均值 - 旧版本均值
        diff = np.mean(sample_t) - np.mean(sample_c)
        diffs.append(diff)
        
    # 计算置信区间 (95% CI)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    mean_diff = np.mean(diffs)
    
    # --- 画图 ---
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 画出 0 线 (红线)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Difference (0)')
    
    # 画出置信区间 (绿线)
    plt.axvline(x=ci_lower, color='green', linestyle=':', linewidth=2, label='95% CI Lower')
    plt.axvline(x=ci_upper, color='green', linestyle=':', linewidth=2, label='95% CI Upper')
    
    plt.title(f'Bootstrap Difference: {metric_col}\n(Test - Control)', fontsize=14)
    plt.xlabel('Difference in Means', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # --- 输出结论 ---
    print(f"--- 结果分析: {metric_col} ---")
    print(f"平均提升量: {mean_diff:.4f}")
    print(f"95% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    if ci_lower > 0:
        print("✅ 结论: 显著提升 (Positive Significance)")
    elif ci_upper < 0:
        print("❌ 结论: 显著下降 (Negative Significance)")
    else:
        print("⚠️ 结论: 无显著差异 (Not Significant) - 区间包含了 0")

# ==========================================
# 2. 这里开始运行你的分析
# ==========================================
df = pd.read_csv(r"C:\Users\HP\Downloads\final-version-data-augmented.csv")

# 筛选 India 用户
df_us = df[df['country'] == 'India']

print("=== India用户 (India Users) 分析 ===")

# 场景 A: 分析时长 (Session Duration)
run_bootstrap_analysis(df_us, 'avg_duration_per_session')

# 场景 B: 分析广告收入 (Ad Revenue)
run_bootstrap_analysis(df_us, 'ad_revenue') 

# 场景 C: 分析页面深度
run_bootstrap_analysis(df_us, 'avg_page_per_session')

# 场景 D: 分析页面频率
run_bootstrap_analysis(df_us, 'session_open')

# 场景 E: 分析广告展示量
run_bootstrap_analysis(df_us, 'ad_impression')

# 场景 F: 分析广告点击量
run_bootstrap_analysis(df_us, 'ad_click')

run_bootstrap_analysis(df_us, 'ad_impression_reward')
run_bootstrap_analysis(df_us, 'rewarded_ratio')
run_bootstrap_analysis(df_us, 'purchase')
run_bootstrap_analysis(df_us, 'ad_click_through_rate')
run_bootstrap_analysis(df_us, 'ad_density')
