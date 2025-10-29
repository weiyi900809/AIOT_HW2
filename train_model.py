#!/usr/bin/env python3
"""
Wine Quality Model Training Script (Enhanced Version)
包含特徵選擇、模型評估和預測區間計算
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Wine Quality Model Training - Enhanced Version")
print("包含特徵選擇與完整模型評估")
print("="*80)

# ============================================================================
# 1. 數據加載
# ============================================================================
print("\n[1] 數據加載...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

try:
    df = pd.read_csv(url, sep=';')
    print(f"✓ 數據集加載成功！共 {df.shape[0]} 筆記錄")
except:
    print("✗ 無法從 UCI 加載，嘗試從本地加載...")
    df = pd.read_csv('winequality-red.csv', sep=';')

# 數據清理
df_clean = df.drop_duplicates()
print(f"✓ 移除重複值後：{df_clean.shape[0]} 筆記錄")

# 分離特徵和目標
X = df_clean.drop('quality', axis=1)
y = df_clean['quality']

print(f"\n特徵數量: {X.shape[1]}")
print(f"特徵名稱: {list(X.columns)}")

# ============================================================================
# 2. 特徵選擇 (Feature Selection)
# ============================================================================
print("\n" + "="*80)
print("[2] 特徵選擇 (Recursive Feature Elimination with Cross-Validation)")
print("="*80)

# 分割數據（用於特徵選擇）
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 標準化（特徵選擇階段）
scaler_temp = StandardScaler()
X_train_scaled_temp = scaler_temp.fit_transform(X_train_temp)
X_test_scaled_temp = scaler_temp.transform(X_test_temp)

# 使用 RFECV 進行特徵選擇（自動選擇最佳特徵數量）
print("\n執行 RFECV 特徵選擇...")
base_model = LinearRegression()
rfecv = RFECV(
    estimator=base_model,
    step=1,
    cv=5,
    scoring='r2',
    min_features_to_select=5
)

rfecv.fit(X_train_scaled_temp, y_train_temp)

print(f"\n✓ 最佳特徵數量: {rfecv.n_features_}")
print(f"✓ 所選特徵: {list(X.columns[rfecv.support_])}")

# 顯示特徵排名
feature_ranking = pd.DataFrame({
    '特徵名稱': X.columns,
    '是否選中': rfecv.support_,
    '排名': rfecv.ranking_
}).sort_values('排名')

print("\n特徵選擇結果:")
print(feature_ranking)

# 儲存特徵選擇資訊
selected_features = X.columns[rfecv.support_].tolist()
joblib.dump(selected_features, 'selected_features.pkl')
joblib.dump(rfecv, 'feature_selector.pkl')

# 視覺化交叉驗證分數
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
         rfecv.cv_results_['mean_test_score'], 
         marker='o', linewidth=2)
plt.xlabel('特徵數量', fontsize=12)
plt.ylabel('交叉驗證 R² 分數', fontsize=12)
plt.title('RFECV: 特徵數量 vs. 模型性能', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--', 
            label=f'最佳特徵數: {rfecv.n_features_}')
plt.legend()
plt.tight_layout()
plt.savefig('feature_selection_cv_scores.png', dpi=300, bbox_inches='tight')
print("\n✓ 特徵選擇圖表已儲存: feature_selection_cv_scores.png")
plt.close()

# ============================================================================
# 3. 使用選定特徵重新訓練模型
# ============================================================================
print("\n" + "="*80)
print("[3] 使用選定特徵重新訓練模型")
print("="*80)

# 只保留選定的特徵
X_selected = X[selected_features]

# 重新分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print(f"\n訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練最終模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n✓ 模型訓練完成！")

# 模型係數
coefficients = pd.DataFrame({
    '特徵': selected_features,
    '係數': model.coef_
}).sort_values(by='係數', key=abs, ascending=False)

print("\n模型係數 (按絕對值排序):")
print(coefficients)

# ============================================================================
# 4. 模型評估 (Model Evaluation)
# ============================================================================
print("\n" + "="*80)
print("[4] 模型評估")
print("="*80)

# 預測
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# 訓練集性能
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print("\n訓練集性能:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  MAE: {train_mae:.4f}")
print(f"  RMSE: {train_rmse:.4f}")

# 測試集性能
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n測試集性能:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  MAE: {test_mae:.4f}")
print(f"  RMSE: {test_rmse:.4f}")

# 交叉驗證
print("\n執行 5 折交叉驗證...")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"交叉驗證 R² 分數: {cv_scores}")
print(f"平均 R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 殘差分析
residuals = y_test - y_test_pred
print("\n殘差統計:")
print(f"  殘差均值: {residuals.mean():.4f}")
print(f"  殘差標準差: {residuals.std():.4f}")
print(f"  最小殘差: {residuals.min():.4f}")
print(f"  最大殘差: {residuals.max():.4f}")

# ============================================================================
# 5. 計算預測區間和信賴區間
# ============================================================================
print("\n" + "="*80)
print("[5] 計算預測區間和信賴區間")
print("="*80)

# 計算標準誤差 (用於預測區間)
mse = mean_squared_error(y_train, y_train_pred)
std_error = np.sqrt(mse)

# 計算信賴區間和預測區間所需的參數
n = len(y_train)
k = X_train_scaled.shape[1]  # 特徵數量
dof = n - k - 1  # 自由度

from scipy import stats
t_value = stats.t.ppf(0.975, dof)  # 95% 信賴水準

print(f"\n標準誤差 (Standard Error): {std_error:.4f}")
print(f"t 值 (95% 信賴水準): {t_value:.4f}")
print(f"自由度: {dof}")

# 計算測試集的預測區間
# 預測區間 = prediction ± t * std_error * sqrt(1 + 1/n + (x-x_mean)^2/sum((x-x_mean)^2))
# 簡化版本: prediction ± t * std_error * sqrt(1 + 1/n)
prediction_interval_width = t_value * std_error * np.sqrt(1 + 1/n)

# 計算信賴區間 (針對均值)
# 信賴區間 = prediction ± t * std_error / sqrt(n)
confidence_interval_width = t_value * std_error / np.sqrt(n)

print(f"\n預測區間寬度 (單側): ±{prediction_interval_width:.4f}")
print(f"信賴區間寬度 (單側): ±{confidence_interval_width:.4f}")

# 儲存區間計算參數
interval_params = {
    'std_error': std_error,
    't_value': t_value,
    'n': n,
    'prediction_interval_width': prediction_interval_width,
    'confidence_interval_width': confidence_interval_width
}
joblib.dump(interval_params, 'interval_params.pkl')

# ============================================================================
# 6. 生成評估圖表
# ============================================================================
print("\n" + "="*80)
print("[6] 生成評估圖表")
print("="*80)

# 創建綜合評估圖
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 圖1: 預測值 vs 實際值 (訓練集)
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, edgecolors='k')
axes[0, 0].plot([y_train.min(), y_train.max()], 
                [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('實際品質', fontsize=11)
axes[0, 0].set_ylabel('預測品質', fontsize=11)
axes[0, 0].set_title(f'訓練集: 預測 vs 實際\nR² = {train_r2:.4f}', 
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 圖2: 預測值 vs 實際值 (測試集)
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('實際品質', fontsize=11)
axes[0, 1].set_ylabel('預測品質', fontsize=11)
axes[0, 1].set_title(f'測試集: 預測 vs 實際\nR² = {test_r2:.4f}', 
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 圖3: 殘差圖
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', color='green')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('預測值', fontsize=11)
axes[1, 0].set_ylabel('殘差', fontsize=11)
axes[1, 0].set_title('殘差圖', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 圖4: 殘差分布
axes[1, 1].hist(residuals, bins=30, edgecolor='k', alpha=0.7, color='purple')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('殘差', fontsize=11)
axes[1, 1].set_ylabel('頻率', fontsize=11)
axes[1, 1].set_title(f'殘差分布\n均值={residuals.mean():.4f}, 標準差={residuals.std():.4f}', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
print("\n✓ 模型評估圖表已儲存: model_evaluation_plots.png")
plt.close()

# 特徵重要性圖
plt.figure(figsize=(10, 6))
coefficients_sorted = coefficients.sort_values('係數')
colors = ['red' if x < 0 else 'green' for x in coefficients_sorted['係數']]
plt.barh(coefficients_sorted['特徵'], coefficients_sorted['係數'], color=colors, alpha=0.7)
plt.xlabel('係數值', fontsize=12)
plt.ylabel('特徵', fontsize=12)
plt.title('特徵重要性 (模型係數)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ 特徵重要性圖表已儲存: feature_importance.png")
plt.close()

# ============================================================================
# 7. 保存模型和相關檔案
# ============================================================================
print("\n" + "="*80)
print("[7] 保存模型和相關檔案")
print("="*80)

joblib.dump(model, 'wine_quality_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(selected_features, 'feature_names.pkl')

# 保存模型評估結果
evaluation_results = {
    'train_r2': train_r2,
    'train_mae': train_mae,
    'train_rmse': train_rmse,
    'test_r2': test_r2,
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'cv_scores': cv_scores,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'coefficients': coefficients.to_dict(),
    'n_features_original': X.shape[1],
    'n_features_selected': len(selected_features),
    'selected_features': selected_features
}
joblib.dump(evaluation_results, 'evaluation_results.pkl')

print("\n✓ 模型已保存為: wine_quality_model.pkl")
print("✓ Scaler 已保存為: feature_scaler.pkl")
print("✓ 特徵名稱已保存為: feature_names.pkl")
print("✓ 評估結果已保存為: evaluation_results.pkl")
print("✓ 區間參數已保存為: interval_params.pkl")
print("✓ 特徵選擇器已保存為: feature_selector.pkl")
print("✓ 選定特徵已保存為: selected_features.pkl")

print("\n" + "="*80)
print("模型訓練與評估完成！")
print("="*80)

# 輸出最終總結
print("\n📊 最終模型總結:")
print(f"  原始特徵數: {X.shape[1]}")
print(f"  選定特徵數: {len(selected_features)}")
print(f"  測試集 R²: {test_r2:.4f}")
print(f"  測試集 MAE: {test_mae:.4f}")
print(f"  測試集 RMSE: {test_rmse:.4f}")
print(f"  交叉驗證 R²: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
