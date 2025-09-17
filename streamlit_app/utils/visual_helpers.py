
"""
Visual helpers and utility functions for Streamlit toll system dashboard
Provides common styling, color schemes, and visualization functions
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Color schemes for consistent styling
COLOR_SCHEMES = {
    'traffic': {
        'free_flow': '#2E8B57',    # Sea Green
        'light': '#FFD700',        # Gold
        'moderate': '#FF8C00',     # Dark Orange
        'heavy': '#DC143C',        # Crimson
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e'     # Orange
    },
    'air_quality': {
        'good': '#2E8B57',         # Sea Green
        'moderate': '#FFD700',     # Gold
        'unhealthy_sensitive': '#FF8C00',  # Dark Orange
        'unhealthy': '#DC143C',    # Crimson
        'very_unhealthy': '#8B0000', # Dark Red
        'hazardous': '#4B0082'     # Indigo
    },
    'revenue': {
        'success': '#2E8B57',      # Sea Green
        'warning': '#FFD700',      # Gold
        'error': '#DC143C',        # Crimson
        'neutral': '#708090'       # Slate Gray
    },
    'system_health': {
        'online': '#2E8B57',       # Sea Green
        'degraded': '#FFD700',     # Gold
        'offline': '#DC143C',      # Crimson
        'maintenance': '#FF8C00'   # Dark Orange
    }
}

# Common layout settings
LAYOUT_CONFIG = {
    'template': 'plotly_white',
    'title_font_size': 16,
    'axis_title_font_size': 12,
    'legend_font_size': 10,
    'height': 400,
    'margin': dict(l=50, r=50, t=50, b=50)
}

def apply_custom_css():
    """Apply custom CSS styling to Streamlit app"""
    st.markdown("""
    <style>
        /* Main content styling */
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
        }
        
        .kpi-value {
            font-size: 2.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .kpi-label {
            font-size: 0.9rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        /* Status indicators */
        .status-good {
            color: #2E8B57;
            font-weight: bold;
        }
        
        .status-warning {
            color: #FFD700;
            font-weight: bold;
        }
        
        .status-error {
            color: #DC143C;
            font-weight: bold;
        }
        
        /* Alert boxes */
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Data tables */
        .dataframe {
            font-size: 0.9rem;
        }
        
        .dataframe th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        /* Sidebar styling */
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #666;
            padding: 2rem 0;
            border-top: 1px solid #eee;
            margin-top: 3rem;
        }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(value, label, delta=None, delta_color="normal"):
    """Create a styled metric card with optional delta"""
    delta_class = f"status-{delta_color}" if delta else ""
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    
    return f"""
    <div class="metric-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """

def create_status_indicator(status, value=None):
    """Create a status indicator with appropriate color"""
    status_map = {
        'online': ('🟢', 'good'),
        'offline': ('🔴', 'error'),
        'degraded': ('🟡', 'warning'),
        'maintenance': ('🟠', 'warning'),
        'good': ('✅', 'good'),
        'moderate': ('⚠️', 'warning'),
        'unhealthy': ('🚨', 'error'),
        'normal': ('🟢', 'good'),
        'high': ('🔴', 'error'),
        'critical': ('💥', 'error')
    }
    
    icon, color_class = status_map.get(status.lower(), ('ℹ️', 'normal'))
    display_text = f"{icon} {status.title()}"
    if value:
        display_text += f" ({value})"
    
    return f'<span class="status-{color_class}">{display_text}</span>'

def create_trend_chart(df, x_col, y_col, title, color_col=None, height=400):
    """Create a standardized trend line chart"""
    if color_col:
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
    else:
        fig = px.line(df, x=x_col, y=y_col, title=title)
    
    fig.update_layout(
        template=LAYOUT_CONFIG['template'],
        height=height,
        title_font_size=LAYOUT_CONFIG['title_font_size'],
        margin=LAYOUT_CONFIG['margin']
    )
    
    return fig

def create_heatmap(data, title="Heatmap", color_scale="RdYlBu_r"):
    """Create a standardized heatmap"""
    fig = px.imshow(
        data,
        title=title,
        color_continuous_scale=color_scale,
        aspect="auto"
    )
    
    fig.update_layout(
        template=LAYOUT_CONFIG['template'],
        height=LAYOUT_CONFIG['height'],
        title_font_size=LAYOUT_CONFIG['title_font_size']
    )
    
    return fig

def create_gauge_chart(value, title, min_val=0, max_val=100, threshold_colors=None):
    """Create a gauge chart for KPIs"""
    if threshold_colors is None:
        threshold_colors = [
            {'range': [min_val, max_val * 0.3], 'color': COLOR_SCHEMES['traffic']['free_flow']},
            {'range': [max_val * 0.3, max_val * 0.7], 'color': COLOR_SCHEMES['traffic']['moderate']},
            {'range': [max_val * 0.7, max_val], 'color': COLOR_SCHEMES['traffic']['heavy']}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': threshold_colors,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.8
            }
        }
    ))
    
    fig.update_layout(
        template=LAYOUT_CONFIG['template'],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_comparison_chart(df, categories, values, title, chart_type='bar'):
    """Create comparison charts (bar, pie, etc.)"""
    if chart_type == 'bar':
        fig = px.bar(
            x=categories,
            y=values,
            title=title,
            labels={'x': 'Category', 'y': 'Value'}
        )
    elif chart_type == 'pie':
        fig = px.pie(
            values=values,
            names=categories,
            title=title
        )
    elif chart_type == 'scatter':
        fig = px.scatter(
            x=categories,
            y=values,
            title=title,
            size=values
        )
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    fig.update_layout(
        template=LAYOUT_CONFIG['template'],
        height=LAYOUT_CONFIG['height'],
        title_font_size=LAYOUT_CONFIG['title_font_size']
    )
    
    return fig

def create_multi_axis_chart(df, x_col, y1_col, y2_col, title, y1_name, y2_name):
    """Create a chart with multiple y-axes"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y1_col],
            mode='lines+markers',
            name=y1_name,
            line=dict(color=COLOR_SCHEMES['traffic']['primary'])
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y2_col],
            mode='lines+markers',
            name=y2_name,
            line=dict(color=COLOR_SCHEMES['traffic']['secondary'])
        ),
        secondary_y=True,
    )
    
    fig.update_xaxis(title_text=x_col.title())
    fig.update_yaxis(title_text=y1_name, secondary_y=False)
    fig.update_yaxis(title_text=y2_name, secondary_y=True)
    
    fig.update_layout(
        title_text=title,
        template=LAYOUT_CONFIG['template'],
        height=LAYOUT_CONFIG['height']
    )
    
    return fig

def create_time_series_with_annotations(df, x_col, y_col, title, annotations=None):
    """Create time series chart with event annotations"""
    fig = px.line(df, x=x_col, y=y_col, title=title)
    
    if annotations:
        for annotation in annotations:
            fig.add_vline(
                x=annotation['x'],
                line_dash="dash",
                line_color=annotation.get('color', 'red'),
                annotation_text=annotation.get('text', ''),
                annotation_position=annotation.get('position', 'top')
            )
    
    fig.update_layout(
        template=LAYOUT_CONFIG['template'],
        height=LAYOUT_CONFIG['height'],
        title_font_size=LAYOUT_CONFIG['title_font_size']
    )
    
    return fig

def format_large_numbers(value):
    """Format large numbers for display"""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.0f}"

def format_currency(value):
    """Format currency values"""
    return f"${value:,.2f}"

def format_percentage(value, decimal_places=1):
    """Format percentage values"""
    return f"{value:.{decimal_places}f}%"

def get_status_color(value, thresholds, reverse=False):
    """Get status color based on thresholds"""
    colors = ['good', 'warning', 'error'] if not reverse else ['error', 'warning', 'good']
    
    if len(thresholds) != 2:
        raise ValueError("Thresholds must contain exactly 2 values")
    
    if value <= thresholds[0]:
        return colors[0]
    elif value <= thresholds[1]:
        return colors[1]
    else:
        return colors[2]

def create_alert_box(message, alert_type='info'):
    """Create styled alert boxes"""
    alert_classes = {
        'success': 'alert-success',
        'warning': 'alert-warning',
        'error': 'alert-danger',
        'danger': 'alert-danger',
        'info': 'alert-info'
    }
    
    class_name = alert_classes.get(alert_type, 'alert-info')
    
    return f'<div class="{class_name}">{message}</div>'

def create_progress_bar(value, max_value=100, color='primary'):
    """Create a progress bar"""
    percentage = min(100, (value / max_value) * 100)
    color_map = {
        'primary': '#007bff',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545'
    }
    
    bar_color = color_map.get(color, '#007bff')
    
    return f"""
    <div style="background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
        <div style="width: {percentage}%; background-color: {bar_color}; height: 20px; 
                    display: flex; align-items: center; justify-content: center; 
                    color: white; font-size: 12px; font-weight: bold;">
            {percentage:.1f}%
        </div>
    </div>
    """

def create_summary_table(data_dict, title="Summary"):
    """Create a formatted summary table"""
    df = pd.DataFrame(list(data_dict.items()), columns=['Metric', 'Value'])
    return df

def apply_conditional_formatting(df, column, thresholds, colors=None):
    """Apply conditional formatting to dataframe columns"""
    if colors is None:
        colors = ['background-color: #d4edda', 'background-color: #fff3cd', 'background-color: #f8d7da']
    
    def color_cells(val):
        if val <= thresholds[0]:
            return colors[0]
        elif val <= thresholds[1]:
            return colors[1]
        else:
            return colors[2]
    
    return df.style.applymap(color_cells, subset=[column])

def create_kpi_dashboard(kpis, cols=4):
    """Create a KPI dashboard layout"""
    kpi_html = '<div style="display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0;">'
    
    for i, (label, value, delta, delta_color) in enumerate(kpis):
        if i % cols == 0 and i > 0:
            kpi_html += '</div><div style="display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0;">'
        
        kpi_html += f'<div style="flex: 1; min-width: 200px;">{create_metric_card(value, label, delta, delta_color)}</div>'
    
    kpi_html += '</div>'
    return kpi_html

def export_chart_config():
    """Return configuration for chart exports"""
    return {
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'toll_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'height': 600,
            'width': 1000,
            'scale': 2
        }
    }

def create_data_quality_report(df, required_columns=None):
    """Generate a data quality report"""
    quality_metrics = {
        'Total Rows': len(df),
        'Total Columns': len(df.columns),
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Data Types': df.dtypes.value_counts().to_dict(),
        'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
    }
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        quality_metrics['Missing Required Columns'] = list(missing_cols) if missing_cols else 'None'
    
    # Check for outliers in numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
    
    quality_metrics['Outliers by Column'] = outliers
    
    return quality_metrics

def generate_insights(df, metric_column, group_column=None):
    """Generate automatic insights from data"""
    insights = []
    
    # Basic statistics
    mean_val = df[metric_column].mean()
    median_val = df[metric_column].median()
    std_val = df[metric_column].std()
    
    insights.append(f"Average {metric_column}: {mean_val:.2f}")
    insights.append(f"Median {metric_column}: {median_val:.2f}")
    
    # Trend analysis
    if 'timestamp' in df.columns or 'date' in df.columns:
        time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df_sorted = df.sort_values(time_col)
        recent_avg = df_sorted.tail(10)[metric_column].mean()
        older_avg = df_sorted.head(10)[metric_column].mean()
        
        if recent_avg > older_avg * 1.1:
            insights.append(f"📈 {metric_column} is trending upward (+{((recent_avg/older_avg - 1) * 100):.1f}%)")
        elif recent_avg < older_avg * 0.9:
            insights.append(f"📉 {metric_column} is trending downward ({((recent_avg/older_avg - 1) * 100):.1f}%)")
        else:
            insights.append(f"➡️ {metric_column} is relatively stable")
    
    # Group analysis
    if group_column and group_column in df.columns:
        group_stats = df.groupby(group_column)[metric_column].mean().sort_values(ascending=False)
        insights.append(f"🏆 Highest {metric_column}: {group_stats.index[0]} ({group_stats.iloc[0]:.2f})")
        insights.append(f"📊 Lowest {metric_column}: {group_stats.index[-1]} ({group_stats.iloc[-1]:.2f})")
    
    # Outlier detection
    Q1 = df[metric_column].quantile(0.25)
    Q3 = df[metric_column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[metric_column] < Q1 - 1.5 * IQR) | (df[metric_column] > Q3 + 1.5 * IQR)]
    
    if len(outliers) > 0:
        insights.append(f"⚠️ {len(outliers)} outliers detected ({(len(outliers)/len(df)*100):.1f}% of data)")
    
    return insights

def create_dashboard_footer():
    """Create a standardized dashboard footer"""
    return f"""
    <div class="footer">
        🚗 Tolling System Dashboard | 
        Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
        <a href="#" onclick="window.location.reload()" style="text-decoration: none;">🔄 Refresh Data</a>
    </div>
    """
