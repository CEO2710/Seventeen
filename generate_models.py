# generate_models.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 创建 saved_models 目录
os.makedirs('saved_models', exist_ok=True)

# 创建与您数据结构匹配的模拟数据
def create_sample_data():
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Sex': np.random.choice([0, 1], n_samples),
        'ASA scores': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
        'tumor location': np.random.choice([1, 2, 3, 4], n_samples),
        'Benign or malignant': np.random.choice([0, 1], n_samples),
        'Admitted to NICU': np.random.choice([0, 1], n_samples),
        'Duration of surgery': np.random.choice([0, 1], n_samples),
        'diabetes': np.random.choice([0, 1], n_samples),
        'CHF': np.random.choice([0, 1], n_samples),
        'Functional dependencies': np.random.choice([0, 1], n_samples),
        'mFI-5': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
        'Type of tumor': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'Unplanned reoperation': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)

# 主函数
def generate_and_save_models():
    print("创建模拟数据集...")
    df = create_sample_data()
    
    print("准备数据...")
    X = df.drop(columns=['Unplanned reoperation'])
    y = df['Unplanned reoperation']
    
    print("创建和保存预处理对象...")
    scaler = StandardScaler()
    scaler.fit(X)  # 在完整数据上拟合
    
    joblib.dump(scaler, 'saved_models/preprocessor.pkl')
    
    print("训练和保存模型...")
    # 使用随机森林作为示例模型
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    
    joblib.dump(model, 'saved_models/best_model.pkl')
    
    print("✅ 模型文件生成成功！")
    print("文件已保存到 saved_models/ 目录：")
    print("  - preprocessor.pkl (数据预处理对象)")
    print("  - best_model.pkl (预测模型)")

if __name__ == "__main__":
    generate_and_save_models()