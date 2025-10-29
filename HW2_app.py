#!/usr/bin/env python3
"""
Wine Quality Prediction - Enhanced Streamlit Web Application
包含特徵選擇、模型評估和預測區間視覺化
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設置頁面配置
st.set_page_config(
    page_title="🍷 紅酒品質預測器 (增強版)",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 載入模型和相關檔案
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """載入所有必要的模型和數據"""
    try:
        model = joblib.load('wine_quality_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        evaluation_results = joblib.load('evaluation_results.pkl')
        interval_params = joblib.load('interval_params.pkl')
        
        return model, scaler, feature_names, evaluation_results, interval_params
    except FileNotFoundError as e:
        st.error(f"❌ 檔案未找到: {e}\n請先執行 train_model.py 訓練模型！")
        st.stop()

model, scaler, feature_names, eval_results, interval_params = load_model_and_data()

# ============================================================================
# 頁面標題
# ============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🍷 紅酒品質預測系統 (增強版)")
    st.subheader("包含特徵選擇與預測區間視覺化")

with col2:
    st.info("💡 使用 {} 個精選特徵進行預測".format(len(feature_names)))

st.markdown("---")

# ============================================================================
# 側邊欄功能選單
# ============================================================================
st.sidebar.markdown("## 🎯 功能選單")
page = st.sidebar.radio(
    "選擇功能:", 
    ["📈 品質預測 (含預測區間)", "🔍 特徵選擇結果", "📊 模型評估", "📉 評估圖表", "ℹ️ 使用說明"]
)

# ============================================================================
# 頁面 1: 品質預測 (含預測區間)
# ============================================================================
if page == "📈 品質預測 (含預測區間)":
    st.header("品質預測與預測區間")
    
    st.markdown(f"""
    ### 請輸入紅酒的理化特性參數
    本模型使用 **{len(feature_names)}** 個精選特徵進行預測
    """)
    
    # 創建特徵字典（用於顯示中文名稱）
    feature_display_names = {
        'fixed acidity': '固定酸度',
        'volatile acidity': '揮發性酸度',
        'citric acid': '檸檬酸',
        'residual sugar': '殘留糖分',
        'chlorides': '氯化物',
        'free sulfur dioxide': '游離二氧化硫',
        'total sulfur dioxide': '總二氧化硫',
        'density': '密度',
        'pH': 'pH值',
        'sulphates': '硫酸鹽',
        'alcohol': '酒精含量'
    }
    
    # 特徵範圍（用於 slider）
    feature_ranges = {
        'fixed acidity': (4.6, 15.9, 7.4, 0.1),
        'volatile acidity': (0.12, 1.58, 0.52, 0.01),
        'citric acid': (0.0, 1.0, 0.26, 0.01),
        'residual sugar': (0.9, 15.5, 2.3, 0.1),
        'chlorides': (0.012, 0.611, 0.087, 0.001),
        'free sulfur dioxide': (1.0, 72.0, 15.9, 0.5),
        'total sulfur dioxide': (6.0, 289.0, 46.5, 0.5),
        'density': (0.9901, 1.0037, 0.9967, 0.0001),
        'pH': (2.74, 4.01, 3.31, 0.01),
        'sulphates': (0.33, 2.0, 0.66, 0.01),
        'alcohol': (8.4, 14.9, 10.4, 0.1)
    }
    
    # 建立輸入表單
    input_values = {}
    
    # 檢查哪些特徵被選中
    cols_per_row = 2
    for idx in range(0, len(feature_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, feature in enumerate(feature_names[idx:idx+cols_per_row]):
            if feature in feature_ranges:
                min_val, max_val, default_val, step = feature_ranges[feature]
                display_name = feature_display_names.get(feature, feature)
                
                with cols[col_idx]:
                    input_values[feature] = st.slider(
                        f"{display_name} ({feature})",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step
                    )
    
    # 準備預測數據
    input_data = np.array([[input_values[feature] for feature in feature_names]])
    
    # 標準化輸入數據
    input_scaled = scaler.transform(input_data)
    
    # 預測按鈕
    if st.button("🔮 預測品質", use_container_width=True, type="primary"):
        # 進行預測
        prediction = model.predict(input_scaled)[0]
        
        # 計算預測區間和信賴區間
        pred_interval_width = interval_params['prediction_interval_width']
        conf_interval_width = interval_params['confidence_interval_width']
        
        pred_lower = prediction - pred_interval_width
        pred_upper = prediction + pred_interval_width
        
        conf_lower = prediction - conf_interval_width
        conf_upper = prediction + conf_interval_width
        
        # 顯示預測結果
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.metric(
            label="🍷 預測品質評分",
            value=f"{prediction:.2f}",
            help="基於多元線性回歸模型的預測"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 顯示區間
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 95% 信賴區間")
            st.info(f"""
            **區間範圍**: [{conf_lower:.2f}, {conf_upper:.2f}]
            
            表示我們有 95% 的信心，真實的**平均品質**落在此區間內。
            """)
        
        with col2:
            st.markdown("### 📈 95% 預測區間")
            st.warning(f"""
            **區間範圍**: [{pred_lower:.2f}, {pred_upper:.2f}]
            
            表示我們有 95% 的信心，**單一樣本**的真實品質落在此區間內。
            """)
        
        # 視覺化預測區間
        st.markdown("### 📉 預測區間視覺化")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 繪製預測點
        ax.scatter([1], [prediction], color='red', s=200, zorder=5, 
                   label=f'預測值: {prediction:.2f}', marker='D')
        
        # 繪製信賴區間
        ax.plot([0.8, 1.2], [conf_lower, conf_lower], 'b-', linewidth=2)
        ax.plot([0.8, 1.2], [conf_upper, conf_upper], 'b-', linewidth=2)
        ax.plot([0.8, 0.8], [conf_lower, conf_upper], 'b-', linewidth=2)
        ax.plot([1.2, 1.2], [conf_lower, conf_upper], 'b-', linewidth=2)
        ax.fill_between([0.8, 1.2], conf_lower, conf_upper, 
                        alpha=0.3, color='blue', label='95% 信賴區間')
        
        # 繪製預測區間
        ax.plot([0.6, 1.4], [pred_lower, pred_lower], 'g--', linewidth=2)
        ax.plot([0.6, 1.4], [pred_upper, pred_upper], 'g--', linewidth=2)
        ax.plot([0.6, 0.6], [pred_lower, pred_upper], 'g--', linewidth=2)
        ax.plot([1.4, 1.4], [pred_lower, pred_upper], 'g--', linewidth=2)
        ax.fill_between([0.6, 1.4], pred_lower, pred_upper, 
                        alpha=0.2, color='green', label='95% 預測區間')
        
        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(max(0, pred_lower - 0.5), min(10, pred_upper + 0.5))
        ax.set_ylabel('品質評分', fontsize=12)
        ax.set_title('預測品質與區間估計', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks([])
        
        st.pyplot(fig)
        plt.close()
        
        # 參數總結
        st.markdown("### 📋 輸入參數總結")
        summary_df = pd.DataFrame({
            '特徵名稱': [feature_display_names.get(f, f) for f in feature_names],
            '英文名稱': feature_names,
            '輸入值': [f"{input_values[f]:.4f}" for f in feature_names]
        })
        st.dataframe(summary_df, use_container_width=True)
        
        # 解釋說明
        st.markdown("""
        ---
        ### 💡 結果解讀
        
        - **預測值**: 模型估計的品質評分
        - **信賴區間**: 針對「平均品質」的估計範圍（較窄）
        - **預測區間**: 針對「單一樣本」的估計範圍（較寬）
        
        預測區間比信賴區間寬，因為它考慮了個體樣本的變異性。
        """)

# ============================================================================
# 頁面 2: 特徵選擇結果
# ============================================================================
elif page == "🔍 特徵選擇結果":
    st.header("特徵選擇結果")
    
    st.markdown(f"""
    ### 📊 特徵選擇統計
    
    - **原始特徵數量**: {eval_results['n_features_original']} 個
    - **選定特徵數量**: {eval_results['n_features_selected']} 個
    - **特徵選擇方法**: RFECV (遞迴特徵消除 + 交叉驗證)
    """)
    
    st.markdown("### ✅ 被選中的特徵")
    selected_df = pd.DataFrame({
        '序號': range(1, len(feature_names) + 1),
        '特徵名稱': feature_names
    })
    st.dataframe(selected_df, use_container_width=True)
    
    # 顯示特徵選擇圖表
    try:
        st.markdown("### 📈 特徵選擇過程")
        st.image('feature_selection_cv_scores.png', 
                 caption='RFECV: 不同特徵數量下的交叉驗證分數',
                 use_container_width=True)
    except:
        st.info("特徵選擇圖表尚未生成，請執行 train_model.py")
    
    # 特徵重要性
    try:
        st.markdown("### 🎯 特徵重要性")
        st.image('feature_importance.png',
                 caption='各特徵的模型係數（表示重要性）',
                 use_container_width=True)
    except:
        st.info("特徵重要性圖表尚未生成")

# ============================================================================
# 頁面 3: 模型評估
# ============================================================================
elif page == "📊 模型評估":
    st.header("模型評估結果")
    
    # 性能指標
    st.markdown("### 📈 模型性能指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="測試集 R²",
            value=f"{eval_results['test_r2']:.4f}",
            help="決定係數，表示模型解釋變異的比例"
        )
    
    with col2:
        st.metric(
            label="測試集 MAE",
            value=f"{eval_results['test_mae']:.4f}",
            help="平均絕對誤差"
        )
    
    with col3:
        st.metric(
            label="測試集 RMSE",
            value=f"{eval_results['test_rmse']:.4f}",
            help="均方根誤差"
        )
    
    with col4:
        st.metric(
            label="交叉驗證 R²",
            value=f"{eval_results['cv_mean']:.4f}",
            delta=f"±{eval_results['cv_std']*2:.4f}",
            help="5折交叉驗證的平均 R² 分數"
        )
    
    # 訓練集 vs 測試集
    st.markdown("### 📊 訓練集 vs 測試集性能")
    
    comparison_df = pd.DataFrame({
        '指標': ['R² Score', 'MAE', 'RMSE'],
        '訓練集': [
            f"{eval_results['train_r2']:.4f}",
            f"{eval_results['train_mae']:.4f}",
            f"{eval_results['train_rmse']:.4f}"
        ],
        '測試集': [
            f"{eval_results['test_r2']:.4f}",
            f"{eval_results['test_mae']:.4f}",
            f"{eval_results['test_rmse']:.4f}"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # 模型係數
    st.markdown("### 🔢 模型係數 (特徵權重)")
    
    if 'coefficients' in eval_results:
        coef_df = pd.DataFrame(eval_results['coefficients'])
        st.dataframe(coef_df, use_container_width=True)
    
    # 交叉驗證分數
    st.markdown("### 🔄 5折交叉驗證分數")
    cv_df = pd.DataFrame({
        '折數': [f'Fold {i+1}' for i in range(len(eval_results['cv_scores']))],
        'R² Score': eval_results['cv_scores']
    })
    st.dataframe(cv_df, use_container_width=True)

# ============================================================================
# 頁面 4: 評估圖表
# ============================================================================
elif page == "📉 評估圖表":
    st.header("模型評估圖表")
    
    # 顯示綜合評估圖
    try:
        st.markdown("### 📊 模型性能綜合評估")
        st.image('model_evaluation_plots.png',
                 caption='包含預測vs實際、殘差圖、殘差分布',
                 use_container_width=True)
    except:
        st.warning("評估圖表尚未生成，請執行 train_model.py")
    
    st.markdown("---")
    
    # 圖表說明
    st.markdown("""
    ### 📖 圖表說明
    
    #### 1. 預測 vs 實際值圖
    - **左上**: 訓練集的預測值與實際值散點圖
    - **右上**: 測試集的預測值與實際值散點圖
    - 理想情況下，點應該落在紅色虛線（y=x）附近
    
    #### 2. 殘差圖 (左下)
    - 顯示預測誤差的分布
    - 理想情況下，殘差應該隨機分布在 0 附近
    
    #### 3. 殘差分布圖 (右下)
    - 顯示殘差的直方圖
    - 應呈現接近常態分布
    """)

# ============================================================================
# 頁面 5: 使用說明
# ============================================================================
elif page == "ℹ️ 使用說明":
    st.header("使用說明")
    
    st.markdown("""
    ## 🚀 快速開始
    
    ### 1. 訓練模型
    ```bash
    python train_model.py
    ```
    
    此步驟將會:
    - 下載紅酒品質資料集
    - 執行特徵選擇 (RFECV)
    - 訓練多元線性回歸模型
    - 計算預測區間和信賴區間參數
    - 生成評估圖表
    - 保存模型檔案
    
    ### 2. 啟動 Web 應用
    ```bash
    streamlit run app.py
    ```
    
    ---
    
    ## 📋 功能說明
    
    ### 📈 品質預測 (含預測區間)
    - 輸入紅酒的理化特性參數
    - 獲得品質評分預測
    - 查看 95% 信賴區間和預測區間
    - 視覺化區間估計
    
    ### 🔍 特徵選擇結果
    - 查看被選中的特徵
    - 了解特徵選擇過程
    - 檢視特徵重要性排序
    
    ### 📊 模型評估
    - 查看詳細的性能指標
    - R², MAE, RMSE 等
    - 交叉驗證結果
    - 模型係數
    
    ### 📉 評估圖表
    - 預測 vs 實際值圖
    - 殘差分析圖
    - 殘差分布圖
    
    ---
    
    ## ❓ 常見問題
    
    ### Q: 信賴區間和預測區間有什麼不同？
    **A**: 
    - **信賴區間**: 估計「平均品質」的範圍（較窄）
    - **預測區間**: 估計「單一樣本」的範圍（較寬）
    
    預測區間考慮了個體變異，因此比信賴區間更寬。
    
    ### Q: 為什麼要進行特徵選擇？
    **A**: 
    - 移除不重要或冗餘的特徵
    - 提高模型泛化能力
    - 減少過擬合風險
    - 提升模型可解釋性
    
    ### Q: RFECV 是什麼？
    **A**: 
    Recursive Feature Elimination with Cross-Validation (遞迴特徵消除 + 交叉驗證)
    - 遞迴地移除不重要的特徵
    - 使用交叉驗證自動選擇最佳特徵數量
    - 避免過度移除特徵
    
    ### Q: 如何判斷模型好壞？
    **A**: 
    - **R² > 0.5**: 模型解釋超過 50% 的變異
    - **MAE < 0.6**: 平均誤差小於 0.6 分
    - **訓練/測試性能接近**: 沒有嚴重過擬合
    
    ---
    
    ## 📚 技術細節
    
    ### 模型類型
    - **多元線性回歸** (Multiple Linear Regression)
    
    ### 特徵選擇
    - **方法**: RFECV
    - **評估指標**: R² Score
    - **交叉驗證**: 5-fold CV
    
    ### 預測區間計算
    ```python
    # 95% 預測區間
    prediction_interval = prediction ± t * std_error * sqrt(1 + 1/n)
    
    # 95% 信賴區間
    confidence_interval = prediction ± t * std_error / sqrt(n)
    ```
    
    其中:
    - `t`: t 分布的臨界值 (95% 信賴水準)
    - `std_error`: 標準誤差
    - `n`: 訓練樣本數
    
    ---
    
    ## 📞 支援
    
    如有問題或建議，歡迎聯繫開發團隊。
    """)

# ============================================================================
# 頁腳
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
<p>🍷 Wine Quality Prediction System (Enhanced) | 包含特徵選擇與預測區間</p>
<p>Last Updated: 2025-10 | Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
