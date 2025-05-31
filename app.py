import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 设置页面
st.set_page_config(page_title="Unplanned Reoperation Risk Prediction", layout="wide")
st.title("Unplanned Reoperation Risk Prediction")

# 加载模型和预处理对象
model = joblib.load('saved_models/best_model.pkl')
scaler = joblib.load('saved_models/preprocessor.pkl')

# 特征描述
feature_descriptions = {
    'Sex': 'Patient sex (0 = Female, 1 = Male)',
    'ASA scores': 'ASA physical status classification',
    'tumor location': 'Location of tumor (1-4)',
    'Benign or malignant': 'Tumor type (0 = Benign, 1 = Malignant)',
    'Admitted to NICU': 'Admission to NICU (0 = No, 1 = Yes)',
    'Duration of surgery': 'Surgery duration (0 = <3 hours, 1 = >=3 hours)',
    'diabetes': 'Diabetes history (0 = No, 1 = Yes)',
    'CHF': 'Congestive heart failure (0 = No, 1 = Yes)',
    'Functional dependencies': 'Functional dependencies (0 = No, 1 = Yes)',
    'mFI-5': 'Modified Frailty Index (0-5)',
    'Type of tumor': 'Tumor classification type (1-5)'
}

# 输入表单
st.sidebar.header("Patient Information")
inputs = {}
for feature in feature_descriptions:
    if feature in ['ASA scores', 'mFI-5']:
        inputs[feature] = st.sidebar.slider(
            f"{feature} ({feature_descriptions[feature]})",
            min_value=0,
            max_value=5,
            value=2
        )
    elif feature == 'tumor location':
        inputs[feature] = st.sidebar.selectbox(
            f"{feature} ({feature_descriptions[feature]})",
            options=[1, 2, 3, 4],
            index=1
        )
    elif feature == 'Type of tumor':
        inputs[feature] = st.sidebar.selectbox(
            f"{feature} ({feature_descriptions[feature]})",
            options=[1, 2, 3, 4, 5],
            index=1
        )
    else:
        inputs[feature] = st.sidebar.radio(
            f"{feature} ({feature_descriptions[feature]})",
            options=[0, 1],
            index=0
        )

# 创建输入数据框
input_df = pd.DataFrame([inputs])

# 显示输入数据
st.subheader("Patient Data Summary")
st.dataframe(input_df)

# 预测按钮
if st.button("Predict Reoperation Risk"):
    # 预处理输入
    input_scaled = scaler.transform(input_df)
    
    # 预测概率
    risk = model.predict_proba(input_scaled)[0][1]
    
    # 显示结果
    st.subheader("Prediction Result")
    st.metric(label="Unplanned Reoperation Risk", value=f"{risk*100:.2f}%")
    
    # 风险可视化
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Risk'], [risk], color='#ff6b6b' if risk > 0.5 else '#4ecdc4', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_title('Reoperation Risk Probability')
    st.pyplot(fig)
    
    # SHAP解释
    st.subheader("Prediction Explanation")
    st.write("The following SHAP values explain how each feature contributed to this prediction:")
    
    # 计算SHAP值
    if isinstance(model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, scaler.transform(input_df))
    else:
        explainer = shap.KernelExplainer(model.predict_proba, scaler.transform(input_df))
    
    shap_values = explainer.shap_values(input_scaled)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification
    
    # 创建SHAP值数据框
    shap_df = pd.DataFrame({
        'Feature': input_df.columns,
        'SHAP Value': shap_values[0],
        'Impact': ['Increases Risk' if val > 0 else 'Decreases Risk' for val in shap_values[0]]
    }).sort_values('SHAP Value', key=abs, ascending=False)
    
    # 显示SHAP值表格
    st.dataframe(shap_df)
    
    # SHAP瀑布图
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explainer(input_scaled)[0], max_display=11, show=False)
    plt.title('Feature Contribution to Prediction', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

# 模型信息和SHAP摘要图
st.sidebar.header("About")
st.sidebar.info("This application predicts the risk of unplanned reoperation using machine learning models.")

st.subheader("Model Explanation")
st.write("The SHAP summary plot below shows the global feature importance across all predictions:")
st.image('shap_summary.png', caption='SHAP Feature Importance Summary')