from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 准备数据：只选 US 用户
df = pd.read_csv(r"C:\Users\HP\Downloads\final-version-data-augmented.csv")

df_tree = df[df['country'] == 'India'].copy()

# 2. 准备特征 (X) 和 目标 (y)
# 我们重点看 page_home，但也放入 duration 和 density 作为辅助
tree_features = ['page_home', 'ad_density', 'avg_duration_per_session']
X = df_tree[tree_features].fillna(0)
y = df_tree['ad_revenue']

# 3. 训练决策树 (Regressor)
# max_depth=3 足够了，太深了看不懂，我们要找的是最顶层的核心阈值
dt_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=42)
dt_model.fit(X, y)

# 4. 打印文字版规则 (Text Representation) - 最直观
print("--- 决策树规则 (US Market) ---")
# feature_names 必须对应 X 的列名
tree_rules = export_text(dt_model, feature_names=tree_features)
print(tree_rules)

# 5. 画图版 (Visual Representation) - 放在报告里好看
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=tree_features, 
          filled=True, 
          rounded=True, 
          fontsize=10,
          precision=4)
plt.title("Decision Tree: Critical Thresholds for US Revenue", fontsize=15)
plt.show()

# --- 验证性画图：散点图 ---
# 让我们亲眼看看这个阈值是不是真的存在
plt.figure(figsize=(8, 5))
sns.regplot(x='page_home', y='ad_revenue', data=df_tree, x_jitter=0.1, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Relationship: Page Home vs. Revenue (US)')
plt.xlabel('Page Home Views')
plt.ylabel('Ad Revenue')
plt.show()