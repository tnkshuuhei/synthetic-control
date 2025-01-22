# scripts/counterfactual.py

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

@dataclass
class SynthControlRequest:
    time_predictors_prior_start: datetime
    time_predictors_prior_end: datetime
    time_optimize_ssr_start: datetime
    time_optimize_ssr_end: datetime
    dependent: str
    treatment_identifier: str
    controls_identifier: List[str]
    predictors: List[str]

@dataclass
class SynthControlResponse:
    weights: Dict[str, float]
    data: List[Dict[str, Any]]

def create_synth_control(df: pd.DataFrame, request: SynthControlRequest) -> SynthControlResponse:
    """合成制御分析を実行する関数"""
    # データの前処理
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # デバッグ情報の出力
    print(f"データセットの大きさ: {df.shape}")
    print(f"ユニークな日付の数: {df['date'].nunique()}")
    print(f"利用可能なメトリクス: {df['metric_key'].unique()}")
    
    # トレーニング期間のマスク
    train_mask = (
        (df['date'] >= request.time_predictors_prior_start) & 
        (df['date'] <= request.time_predictors_prior_end)
    )
    
    # トレーニングデータの件数を確認
    print(f"トレーニングデータの件数: {train_mask.sum()}")
    
    # 処理用の配列を初期化
    feature_matrix = []
    control_names = []
    
    # 各制御群について特徴量を収集
    for control in request.controls_identifier:
        control_features = []
        for predictor in request.predictors:
            predictor_values = df[
                (df['metric_key'] == predictor) & 
                (df['origin_key'] == control) &
                train_mask
            ]['value'].values
            
            if len(predictor_values) > 0:
                control_features.append(predictor_values)
                
        if len(control_features) == len(request.predictors):
            feature_matrix.append(np.mean(control_features, axis=0))
            control_names.append(control)
    
    # 処置群のデータを収集
    treatment_values = []
    for predictor in request.predictors:
        pred_values = df[
            (df['metric_key'] == predictor) & 
            (df['origin_key'] == request.treatment_identifier) &
            train_mask
        ]['value'].values
        
        if len(pred_values) > 0:
            treatment_values.append(pred_values)
    
    # データの形状を確認
    if not feature_matrix or not treatment_values:
        raise ValueError("データが不十分です")
    
    # データを行列形式に変換
    X_control = np.array(feature_matrix).T
    y_treatment = np.mean(treatment_values, axis=0)
    
    print(f"特徴量行列の形状: {X_control.shape}")
    print(f"目的変数の形状: {y_treatment.shape}")
    
    # スケーリング
    scaler = StandardScaler()
    X_control_scaled = scaler.fit_transform(X_control)
    y_treatment_scaled = scaler.fit_transform(y_treatment.reshape(-1, 1)).ravel()
    
    # モデルの学習
    model = LassoCV(positive=True, cv=5)
    model.fit(X_control_scaled, y_treatment_scaled)
    
    # 重みの計算
    weights = {}
    for control, weight in zip(control_names, model.coef_):
        if weight > 0.001:
            weights[control] = float(weight)
    
    # 重みの正規化
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    # 予測データの準備
    prediction_data = []
    dates = sorted(df['date'].unique())
    
    for date in dates:
        treatment_value = df[
            (df['metric_key'] == request.dependent) & 
            (df['origin_key'] == request.treatment_identifier) & 
            (df['date'] == date)
        ]['value'].iloc[0] if len(df[
            (df['metric_key'] == request.dependent) & 
            (df['origin_key'] == request.treatment_identifier) & 
            (df['date'] == date)
        ]) > 0 else 0
        
        synthetic_value = 0
        for control, weight in weights.items():
            control_value = df[
                (df['metric_key'] == request.dependent) & 
                (df['origin_key'] == control) & 
                (df['date'] == date)
            ]['value'].iloc[0] if len(df[
                (df['metric_key'] == request.dependent) & 
                (df['origin_key'] == control) & 
                (df['date'] == date)
            ]) > 0 else 0
            synthetic_value += weight * control_value
        
        prediction_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'treatment': float(treatment_value),
            'synthetic': float(synthetic_value)
        })
    
    return SynthControlResponse(weights=weights, data=prediction_data)