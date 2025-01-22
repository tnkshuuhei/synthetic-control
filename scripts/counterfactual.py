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
    """
    Synthetic Control分析を実行する関数
    """
    df['date'] = pd.to_datetime(df['date'])
    
    train_mask = (
        (df['date'] >= request.time_predictors_prior_start) & 
        (df['date'] <= request.time_predictors_prior_end)
    )
    
    treatment_data = df[df['origin_key'] == request.treatment_identifier]
    control_data = df[df['origin_key'].isin(request.controls_identifier)]
    
    X_control = []
    y_treatment = []
    
    for predictor in request.predictors:
        treatment_values = treatment_data[train_mask][
            treatment_data['metric_key'] == predictor
        ]['value'].values
        y_treatment.extend(treatment_values)
        
        for control in request.controls_identifier:
            control_values = control_data[train_mask][
                (control_data['origin_key'] == control) & 
                (control_data['metric_key'] == predictor)
            ]['value'].values
            X_control.append(control_values)
    
    X_control = np.array(X_control).T
    y_treatment = np.array(y_treatment)
    
    scaler = StandardScaler()
    X_control_scaled = scaler.fit_transform(X_control)
    y_treatment_scaled = scaler.fit_transform(y_treatment.reshape(-1, 1)).ravel()
    
    model = LassoCV(positive=True, cv=5)
    model.fit(X_control_scaled, y_treatment_scaled)
    
    weights = {}
    for control, weight in zip(request.controls_identifier, model.coef_):
        if weight > 0.001:  # Remove small weights
            weights[control] = float(weight)
    
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    prediction_data = []
    dates = sorted(df['date'].unique())
    
    for date in dates:
        actual = float(treatment_data[
            (treatment_data['date'] == date) & 
            (treatment_data['metric_key'] == request.dependent)
        ]['value'].iloc[0]) if len(treatment_data[
            (treatment_data['date'] == date) & 
            (treatment_data['metric_key'] == request.dependent)
        ]) > 0 else 0
        
        synthetic = 0
        for control, weight in weights.items():
            control_value = float(control_data[
                (control_data['date'] == date) & 
                (control_data['origin_key'] == control) & 
                (control_data['metric_key'] == request.dependent)
            ]['value'].iloc[0]) if len(control_data[
                (control_data['date'] == date) & 
                (control_data['origin_key'] == control) & 
                (control_data['metric_key'] == request.dependent)
            ]) > 0 else 0
            synthetic += weight * control_value
        
        prediction_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'treatment': actual,
            'synthetic': synthetic
        })
    
    return SynthControlResponse(weights=weights, data=prediction_data)