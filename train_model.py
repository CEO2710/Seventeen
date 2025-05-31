import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_excel('2222222.xlsx', sheet_name='Sheet1')

# 定义特征和目标变量
X = df.drop(columns=['Unplanned reoperation'])
y = df['Unplanned reoperation']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存预处理对象
joblib.dump(scaler, 'saved_models/preprocessor.pkl')

# 初始化模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# 训练和评估模型
results = {}
best_auc = 0
best_model = None

for name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测概率
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # 计算指标
    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # 保存结果
    results[name] = {
        'AUC': auc,
        'Accuracy': accuracy,
        'F1': f1,
        'Recall': recall,
        'Precision': precision
    }
    
    # 更新最佳模型
    if auc > best_auc:
        best_auc = auc
        best_model = model

# 保存最佳模型
joblib.dump(best_model, 'saved_models/best_model.pkl')

# 打印模型比较结果
print("Model Comparison Results:")
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# 使用SHAP解释最佳模型
if isinstance(best_model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
    explainer = shap.TreeExplainer(best_model)
elif isinstance(best_model, LogisticRegression):
    explainer = shap.LinearExplainer(best_model, X_train_scaled)
else:
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train_scaled)

# 计算SHAP值
shap_values = explainer.shap_values(X_test_scaled)

# 绘制并保存SHAP摘要图
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # 对于二分类问题取正类的SHAP值

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar", show=False)
plt.title('SHAP Feature Importance', fontsize=16)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300)
plt.close()

print(f"Best model saved: {type(best_model).__name__} with AUC: {best_auc:.4f}")