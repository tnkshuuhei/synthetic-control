import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import pandas as pd

def make_linechart(
    df: pd.DataFrame,
    title: str,
    treatment_label: str = "Treatment",
    events: List[Dict] = None
) -> go.Figure:
    """時系列データの可視化関数"""
    fig = px.line(
        df,
        x='date',
        y='value',
        color='cohort',
        title=title,
        labels={'value': 'Value', 'date': 'Date', 'cohort': 'Group'}
    )
    
    fig.update_traces(
        line=dict(width=2),
        selector=dict(name='treatment')
    )
    
    if events:
        for event in events:
            fig.add_vline(
                x=event['date'],
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )
            fig.add_annotation(
                x=event['date'],
                y=1.05,
                yref="paper",
                text=event['text'],
                showarrow=False,
                textangle=-90
            )
    
    fig.update_layout(
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig