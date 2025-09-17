
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Traffic Prediction", page_icon="🔮", layout="wide")

st.title("🔮 AI-Powered Traffic & Revenue Prediction")
st.markdown("Machine learning forecasts for traffic patterns, revenue optimization, and capacity planning")

# Generate comprehensive historical data for ML training
@st.cache_data
def generate_historical_data():
    np.random.seed(42)
    
    # Generate 6 months of hourly data for training
    dates = pd.date_range(start=datetime.now() - timedelta(days=180), 
                         end=datetime.now(), freq='H')
    
    data = []
    
    for date in dates:
        hour = date.hour
        weekday = date.weekday()
        month = date.month
        is_weekend = weekday >= 5
        is_holiday = np.random.random() < 0.02  # 2% chance of holiday
        
        # Seasonal patterns
        seasonal_multiplier = 1.0
        if month in [6, 7, 8]:  # Summer - more travel
            seasonal_multiplier = 1.3
        elif month in [12, 1]:  # Winter - less travel
            seasonal_multiplier = 0.8
        elif month in [3, 4]:  # Spring - moderate increase
            seasonal_multiplier = 1.1
        
        # Weather simulation
        weather = np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], 
                                 p=[0.6, 0.25, 0.05, 0.1])
        weather_multiplier = {
            'Clear': 1.0,
            'Rain': 0.8,
            'Snow': 0.6,
            'Fog': 0.7
        }[weather]
        
        # Traffic patterns
        if is_weekend:
            if 10 <= hour <= 18:  # Weekend leisure traffic
                base_traffic = 120 * seasonal_multiplier
            elif 19 <= hour <= 22:  # Weekend evening
                base_traffic = 100 * seasonal_multiplier
            else:
                base_traffic = 30 * seasonal_multiplier
        else:  # Weekday
            if 7 <= hour <= 9:  # Morning rush
                base_traffic = 200 * seasonal_multiplier
            elif 17 <= hour <= 19:  # Evening rush
                base_traffic = 220 * seasonal_multiplier
            elif 12 <= hour <= 14:  # Lunch
                base_traffic = 80 * seasonal_multiplier
            elif 10 <= hour <= 16:  # Daytime
                base_traffic = 60 * seasonal_multiplier
            else:  # Night/early morning
                base_traffic = 20 * seasonal_multiplier
        
        # Special events (random)
        event_multiplier = 1.0
        if np.random.random() < 0.01:  # 1% chance of special event
            event_multiplier = np.random.uniform(1.5, 3.0)
        
        if is_holiday:
            if weekday < 5:  # Holiday on weekday
                base_traffic *= 0.3  # Much less traffic
            else:
                base_traffic *= 1.2  # Slightly more weekend traffic
        
        # Apply all multipliers
        final_traffic = base_traffic * weather_multiplier * event_multiplier
        vehicle_count = max(0, int(np.random.poisson(final_traffic)))
        
        # Revenue calculation
        vehicle_types = np.random.choice(['Car', 'Truck', 'Bus', 'Motorcycle'], 
                                       size=vehicle_count,
                                       p=[0.75, 0.15, 0.05, 0.05])
        
        toll_rates = {'Car': 3.50, 'Truck': 12.00, 'Bus': 8.50, 'Motorcycle': 2.00}
        
        # Peak pricing
        peak_multiplier = 1.0
        if not is_weekend and (7 <= hour <= 9 or 17 <= hour <= 19):
            peak_multiplier = 1.5
        
        hourly_revenue = sum([toll_rates[vtype] * peak_multiplier for vtype in vehicle_types])
        
        # Additional features for ML
        data.append({
            'timestamp': date,
            'vehicle_count': vehicle_count,
            'revenue': hourly_revenue,
            'hour': hour,
            'weekday': weekday,
            'month': month,
            'is_weekend': int(is_weekend),
            'is_holiday': int(is_holiday),
            'weather': weather,
            'weather_score': weather_multiplier,
            'seasonal_multiplier': seasonal_multiplier,
            'temperature': np.random.normal(20, 10),
            'is_rush_hour': int((7 <= hour <= 9) or (17 <= hour <= 19)),
            'day_of_year': date.timetuple().tm_yday,
            'week_of_year': date.isocalendar()[1]
        })
    
    return pd.DataFrame(data)

# Load historical data
@st.cache_data
def load_and_prepare_data():
    df = generate_historical_data()
    
    # Feature engineering
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Weather encoding
    weather_encoding = {'Clear': 0, 'Rain': 1, 'Snow': 2, 'Fog': 3}
    df['weather_encoded'] = df['weather'].map(weather_encoding)
    
    # Lag features
    df['traffic_lag_1h'] = df['vehicle_count'].shift(1)
    df['traffic_lag_24h'] = df['vehicle_count'].shift(24)
    df['traffic_lag_168h'] = df['vehicle_count'].shift(168)  # 1 week
    
    df['revenue_lag_1h'] = df['revenue'].shift(1)
    df['revenue_lag_24h'] = df['revenue'].shift(24)
    df['revenue_lag_168h'] = df['revenue'].shift(168)
    
    # Rolling averages
    df['traffic_ma_24h'] = df['vehicle_count'].rolling(24, min_periods=1).mean()
    df['traffic_ma_168h'] = df['vehicle_count'].rolling(168, min_periods=1).mean()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

# Train ML models
@st.cache_data
def train_models(df):
    # Features for prediction
    feature_columns = [
        'hour', 'weekday', 'month', 'is_weekend', 'is_holiday', 'weather_encoded',
        'temperature', 'is_rush_hour', 'seasonal_multiplier',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'traffic_lag_1h', 'traffic_lag_24h', 'traffic_lag_168h',
        'traffic_ma_24h', 'traffic_ma_168h'
    ]
    
    revenue_feature_columns = feature_columns + [
        'revenue_lag_1h', 'revenue_lag_24h', 'revenue_lag_168h'
    ]
    
    X_traffic = df[feature_columns]
    y_traffic = df['vehicle_count']
    
    X_revenue = df[revenue_feature_columns + ['vehicle_count']]
    y_revenue = df['revenue']
    
    # Train traffic prediction model
    traffic_model = RandomForestRegressor(n_estimators=100, random_state=42)
    traffic_model.fit(X_traffic, y_traffic)
    
    # Train revenue prediction model
    revenue_model = RandomForestRegressor(n_estimators=100, random_state=42)
    revenue_model.fit(X_revenue, y_revenue)
    
    # Feature importance
    traffic_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': traffic_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    revenue_importance = pd.DataFrame({
        'feature': revenue_feature_columns + ['vehicle_count'],
        'importance': revenue_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return traffic_model, revenue_model, traffic_importance, revenue_importance

# Generate predictions
def generate_predictions(traffic_model, revenue_model, df, hours_ahead=72):
    """Generate predictions for the next N hours"""
    
    # Get the last known values
    last_row = df.iloc[-1].copy()
    predictions = []
    
    for hour_offset in range(1, hours_ahead + 1):
        # Calculate future timestamp
        future_time = last_row['timestamp'] + timedelta(hours=hour_offset)
        
        # Create features for prediction
        pred_features = {
            'hour': future_time.hour,
            'weekday': future_time.weekday(),
            'month': future_time.month,
            'is_weekend': int(future_time.weekday() >= 5),
            'is_holiday': 0,  # Simplified - assume no holidays
            'weather_encoded': 0,  # Assume clear weather
            'temperature': 20,  # Default temperature
            'is_rush_hour': int((7 <= future_time.hour <= 9) or (17 <= future_time.hour <= 19)),
            'seasonal_multiplier': 1.0,  # Simplified
            'hour_sin': np.sin(2 * np.pi * future_time.hour / 24),
            'hour_cos': np.cos(2 * np.pi * future_time.hour / 24),
            'day_sin': np.sin(2 * np.pi * future_time.timetuple().tm_yday / 365),
            'day_cos': np.cos(2 * np.pi * future_time.timetuple().tm_yday / 365),
        }
        
        # Use recent data for lag features
        if hour_offset == 1:
            pred_features['traffic_lag_1h'] = last_row['vehicle_count']
            pred_features['traffic_lag_24h'] = df.iloc[-24]['vehicle_count'] if len(df) >= 24 else last_row['vehicle_count']
            pred_features['traffic_lag_168h'] = df.iloc[-168]['vehicle_count'] if len(df) >= 168 else last_row['vehicle_count']
            pred_features['traffic_ma_24h'] = df['vehicle_count'].tail(24).mean()
            pred_features['traffic_ma_168h'] = df['vehicle_count'].tail(168).mean() if len(df) >= 168 else df['vehicle_count'].mean()
        else:
            # Use predicted values for lag features
            pred_features['traffic_lag_1h'] = predictions[-1]['predicted_traffic'] if predictions else last_row['vehicle_count']
            pred_features['traffic_lag_24h'] = predictions[-24]['predicted_traffic'] if len(predictions) >= 24 else last_row['vehicle_count']
            pred_features['traffic_lag_168h'] = last_row['vehicle_count']  # Simplified
            pred_features['traffic_ma_24h'] = np.mean([p['predicted_traffic'] for p in predictions[-23:]] + [last_row['vehicle_count']]) if len(predictions) >= 23 else last_row['vehicle_count']
            pred_features['traffic_ma_168h'] = last_row['vehicle_count']  # Simplified
        
        # Predict traffic
        X_traffic = np.array([[pred_features[col] for col in [
            'hour', 'weekday', 'month', 'is_weekend', 'is_holiday', 'weather_encoded',
            'temperature', 'is_rush_hour', 'seasonal_multiplier',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'traffic_lag_1h', 'traffic_lag_24h', 'traffic_lag_168h',
            'traffic_ma_24h', 'traffic_ma_168h'
        ]]])
        
        predicted_traffic = max(0, traffic_model.predict(X_traffic)[0])
        
        # Predict revenue
        revenue_features = pred_features.copy()
        revenue_features['vehicle_count'] = predicted_traffic
        revenue_features['revenue_lag_1h'] = last_row['revenue'] if hour_offset == 1 else (predictions[-1]['predicted_revenue'] if predictions else last_row['revenue'])
        revenue_features['revenue_lag_24h'] = last_row['revenue']  # Simplified
        revenue_features['revenue_lag_168h'] = last_row['revenue']  # Simplified
        
        X_revenue = np.array([[revenue_features[col] for col in [
            'hour', 'weekday', 'month', 'is_weekend', 'is_holiday', 'weather_encoded',
            'temperature', 'is_rush_hour', 'seasonal_multiplier',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'traffic_lag_1h', 'traffic_lag_24h', 'traffic_lag_168h',
            'traffic_ma_24h', 'traffic_ma_168h', 'revenue_lag_1h', 
            'revenue_lag_24h', 'revenue_lag_168h', 'vehicle_count'
        ]]])
        
        predicted_revenue = max(0, revenue_model.predict(X_revenue)[0])
        
        predictions.append({
            'timestamp': future_time,
            'predicted_traffic': predicted_traffic,
            'predicted_revenue': predicted_revenue,
            'hour': future_time.hour,
            'weekday': future_time.weekday(),
            'is_weekend': future_time.weekday() >= 5
        })
    
    return pd.DataFrame(predictions)

# Load data and train models
df = load_and_prepare_data()
traffic_model, revenue_model, traffic_importance, revenue_importance = train_models(df)

# Sidebar controls
st.sidebar.header("🔮 Prediction Settings")

# Prediction horizon
prediction_hours = st.sidebar.slider(
    "Prediction Horizon (hours)",
    min_value=6,
    max_value=168,  # 1 week
    value=72,  # 3 days
    step=6
)

# Scenario analysis
st.sidebar.subheader("📊 Scenario Analysis")
weather_scenario = st.sidebar.selectbox(
    "Weather Condition",
    options=['Clear', 'Rain', 'Snow', 'Fog'],
    index=0
)

special_event = st.sidebar.checkbox("Special Event Expected", False)
event_multiplier = st.sidebar.slider(
    "Event Traffic Multiplier",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    disabled=not special_event
) if special_event else 1.0

# Generate predictions
predictions_df = generate_predictions(traffic_model, revenue_model, df, prediction_hours)

# Apply scenario adjustments
weather_multipliers = {'Clear': 1.0, 'Rain': 0.8, 'Snow': 0.6, 'Fog': 0.7}
weather_mult = weather_multipliers[weather_scenario]

predictions_df['predicted_traffic_adjusted'] = predictions_df['predicted_traffic'] * weather_mult * event_multiplier
predictions_df['predicted_revenue_adjusted'] = predictions_df['predicted_revenue'] * weather_mult * event_multiplier

# Dashboard layout
st.subheader("🎯 Prediction Summary")

# Key prediction metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    next_24h_traffic = predictions_df.head(24)['predicted_traffic_adjusted'].sum()
    st.metric("Next 24h Traffic", f"{int(next_24h_traffic):,} vehicles")

with col2:
    next_24h_revenue = predictions_df.head(24)['predicted_revenue_adjusted'].sum()
    st.metric("Next 24h Revenue", f"${next_24h_revenue:,.0f}")

with col3:
    peak_hour = predictions_df.loc[predictions_df['predicted_traffic_adjusted'].idxmax()]
    st.metric("Peak Traffic Hour", f"{peak_hour['timestamp'].strftime('%m/%d %H:00')}")

with col4:
    avg_hourly_revenue = predictions_df['predicted_revenue_adjusted'].mean()
    st.metric("Avg Hourly Revenue", f"${avg_hourly_revenue:.0f}")

# Main prediction charts
st.subheader("📈 Traffic & Revenue Forecasts")

tab1, tab2, tab3 = st.tabs(["Traffic Forecast", "Revenue Forecast", "Combined View"])

with tab1:
    # Traffic prediction with historical context
    recent_history = df.tail(168)  # Last week
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=recent_history['timestamp'],
        y=recent_history['vehicle_count'],
        mode='lines',
        name='Historical Traffic',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions_df['timestamp'],
        y=predictions_df['predicted_traffic_adjusted'],
        mode='lines',
        name='Predicted Traffic',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add vertical line at current time
    fig.add_vline(
        x=df['timestamp'].iloc[-1],
        line_dash="dot",
        line_color="green",
        annotation_text="Current Time"
    )
    
    fig.update_layout(
        title="Traffic Volume Prediction",
        xaxis_title="Time",
        yaxis_title="Vehicles per Hour",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Revenue prediction
    fig = go.Figure()
    
    # Historical revenue
    fig.add_trace(go.Scatter(
        x=recent_history['timestamp'],
        y=recent_history['revenue'],
        mode='lines',
        name='Historical Revenue',
        line=dict(color='green', width=2)
    ))
    
    # Predicted revenue
    fig.add_trace(go.Scatter(
        x=predictions_df['timestamp'],
        y=predictions_df['predicted_revenue_adjusted'],
        mode='lines',
        name='Predicted Revenue',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    fig.add_vline(
        x=df['timestamp'].iloc[-1],
        line_dash="dot",
        line_color="green",
        annotation_text="Current Time"
    )
    
    fig.update_layout(
        title="Revenue Prediction",
        xaxis_title="Time",
        yaxis_title="Revenue per Hour ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Combined traffic and revenue view
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Traffic
    fig.add_trace(
        go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['predicted_traffic_adjusted'],
            mode='lines',
            name='Predicted Traffic',
            line=dict(color='blue')
        ),
        secondary_y=False,
    )
    
    # Revenue
    fig.add_trace(
        go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['predicted_revenue_adjusted'],
            mode='lines',
            name='Predicted Revenue',
            line=dict(color='green')
        ),
        secondary_y=True,
    )
    
    fig.update_xaxis(title_text="Time")
    fig.update_yaxis(title_text="Traffic (vehicles/hour)", secondary_y=False)
    fig.update_yaxis(title_text="Revenue ($/hour)", secondary_y=True)
    fig.update_layout(title_text="Combined Traffic & Revenue Forecast", height=400)
    
    st.plotly_chart(fig, use_container_width=True)

# Pattern analysis
st.subheader("🔄 Pattern Analysis")

col1, col2 = st.columns(2)

with col1:
    # Hourly patterns
    hourly_patterns = predictions_df.groupby('hour').agg({
        'predicted_traffic_adjusted': 'mean',
        'predicted_revenue_adjusted': 'mean'
    }).reset_index()
    
    fig = px.line(
        hourly_patterns,
        x='hour',
        y='predicted_traffic_adjusted',
        title="Predicted Hourly Traffic Pattern",
        labels={'hour': 'Hour of Day', 'predicted_traffic_adjusted': 'Average Traffic'}
    )
    fig.update_layout(xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Weekly patterns
    weekday_patterns = predictions_df.groupby('weekday').agg({
        'predicted_traffic_adjusted': 'mean',
        'predicted_revenue_adjusted': 'mean'
    }).reset_index()
    
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_patterns['weekday_name'] = weekday_patterns['weekday'].map(dict(enumerate(weekday_names)))
    
    fig = px.bar(
        weekday_patterns,
        x='weekday_name',
        y='predicted_traffic_adjusted',
        title="Predicted Daily Traffic Pattern",
        labels={'weekday_name': 'Day of Week', 'predicted_traffic_adjusted': 'Average Traffic'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Model performance and insights
st.subheader("🧠 Model Insights")

col1, col2 = st.columns(2)

with col1:
    # Feature importance for traffic prediction
    fig = px.bar(
        traffic_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Features for Traffic Prediction",
        labels={'importance': 'Feature Importance', 'feature': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Feature importance for revenue prediction
    fig = px.bar(
        revenue_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Features for Revenue Prediction",
        labels={'importance': 'Feature Importance', 'feature': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Capacity planning
st.subheader("⚙️ Capacity Planning Insights")

# Calculate capacity utilization
max_hourly_capacity = 1000  # vehicles per hour (assumed)
capacity_analysis = predictions_df.copy()
capacity_analysis['capacity_utilization'] = (capacity_analysis['predicted_traffic_adjusted'] / max_hourly_capacity * 100)

col1, col2, col3 = st.columns(3)

with col1:
    # Hours over capacity
    over_capacity_hours = len(capacity_analysis[capacity_analysis['capacity_utilization'] > 80])
    st.metric("High Utilization Hours (>80%)", over_capacity_hours)
    
    max_utilization = capacity_analysis['capacity_utilization'].max()
    st.metric("Peak Capacity Utilization", f"{max_utilization:.1f}%")

with col2:
    # Revenue optimization opportunities
    low_revenue_hours = len(predictions_df[predictions_df['predicted_revenue_adjusted'] < 100])
    st.metric("Low Revenue Hours (<$100)", low_revenue_hours)
    
    revenue_variance = predictions_df['predicted_revenue_adjusted'].std()
    st.metric("Revenue Variability", f"${revenue_variance:.0f}")

with col3:
    # Weekend vs weekday comparison
    weekend_avg = predictions_df[predictions_df['is_weekend']]['predicted_traffic_adjusted'].mean()
    weekday_avg = predictions_df[~predictions_df['is_weekend']]['predicted_traffic_adjusted'].mean()
    weekend_diff = ((weekend_avg / weekday_avg) - 1) * 100 if weekday_avg > 0 else 0
    
    st.metric("Weekend vs Weekday Traffic", f"{weekend_diff:+.1f}%")

# Capacity utilization over time
fig = px.line(
    capacity_analysis,
    x='timestamp',
    y='capacity_utilization',
    title="Predicted Capacity Utilization Over Time",
    labels={'capacity_utilization': 'Capacity Utilization (%)', 'timestamp': 'Time'}
)
fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="High Utilization Threshold")
fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Maximum Capacity")
st.plotly_chart(fig, use_container_width=True)

# Recommendations based on predictions
st.subheader("💡 AI-Generated Recommendations")

recommendations = []

# Analyze predicted patterns
peak_hours = capacity_analysis[capacity_analysis['capacity_utilization'] > 80]
if not peak_hours.empty:
    peak_times = peak_hours.groupby('hour').size().sort_values(ascending=False)
    top_peak_hour = peak_times.index[0]
    recommendations.append(f"🔴 **Capacity Alert**: Hour {top_peak_hour} consistently shows high utilization. Consider dynamic pricing or additional lane opening.")

# Revenue optimization
low_revenue_periods = predictions_df[predictions_df['predicted_revenue_adjusted'] < predictions_df['predicted_revenue_adjusted'].quantile(0.25)]
if not low_revenue_periods.empty:
    low_revenue_hours = low_revenue_periods['hour'].value_counts().head(3).index.tolist()
    recommendations.append(f"💰 **Revenue Optimization**: Hours {', '.join(map(str, low_revenue_hours))} show lower revenue potential. Consider promotional pricing or targeted marketing.")

# Traffic distribution
if weekend_diff > 20:
    recommendations.append("📊 **Weekend Traffic**: Significantly higher weekend traffic predicted. Ensure adequate staffing and consider weekend-specific pricing strategies.")
elif weekend_diff < -20:
    recommendations.append("📊 **Weekday Focus**: Weekday traffic dominates. Consider maintenance scheduling during low weekend periods.")

# Weather impact
if weather_scenario != 'Clear':
    recommendations.append(f"🌦️ **Weather Preparation**: {weather_scenario} conditions predicted to reduce traffic by {(1-weather_mult)*100:.0f}%. Prepare for revenue impact and adjust operations accordingly.")

# Special events
if special_event:
    recommendations.append(f"🎉 **Event Management**: Special event expected to increase traffic by {(event_multiplier-1)*100:.0f}%. Increase staffing, prepare for congestion, and optimize pricing strategies.")

# Long-term trends
traffic_trend = (predictions_df.tail(24)['predicted_traffic_adjusted'].mean() / 
                predictions_df.head(24)['predicted_traffic_adjusted'].mean() - 1) * 100
if traffic_trend > 5:
    recommendations.append(f"📈 **Growing Demand**: Traffic trending upward by {traffic_trend:.1f}%. Consider capacity expansion planning.")
elif traffic_trend < -5:
    recommendations.append(f"📉 **Declining Demand**: Traffic trending downward by {abs(traffic_trend):.1f}%. Review pricing strategies and service offerings.")

if len(recommendations) == 0:
    recommendations.append("✅ **Optimal Conditions**: Traffic and revenue patterns appear stable. Continue monitoring for any changes.")

for rec in recommendations:
    st.markdown(rec)

# Detailed predictions table
st.subheader("📋 Detailed Predictions")

# Show next 48 hours with key metrics
detail_predictions = predictions_df.head(48).copy()
detail_predictions['timestamp_formatted'] = detail_predictions['timestamp'].dt.strftime('%m/%d %H:00')
detail_predictions['capacity_util'] = (detail_predictions['predicted_traffic_adjusted'] / max_hourly_capacity * 100)

display_predictions = detail_predictions[[
    'timestamp_formatted', 'predicted_traffic_adjusted', 'predicted_revenue_adjusted', 'capacity_util'
]].round(0)

st.dataframe(
    display_predictions,
    use_container_width=True,
    column_config={
        'timestamp_formatted': st.column_config.TextColumn('Time'),
        'predicted_traffic_adjusted': st.column_config.NumberColumn('Traffic', format="%d vehicles"),
        'predicted_revenue_adjusted': st.column_config.NumberColumn('Revenue', format="$%.0f"),
        'capacity_util': st.column_config.ProgressColumn('Capacity %', min_value=0, max_value=100)
    }
)

# Export predictions
st.subheader("📤 Export Predictions")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Predictions"):
        export_df = predictions_df[[
            'timestamp', 'predicted_traffic_adjusted', 'predicted_revenue_adjusted'
        ]].copy()
        export_df.columns = ['timestamp', 'predicted_traffic', 'predicted_revenue']
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"traffic_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Generate Prediction Report"):
        report = f"""
        # Traffic & Revenue Prediction Report
        
        **Prediction Horizon**: {prediction_hours} hours
        **Weather Scenario**: {weather_scenario}
        **Special Event**: {'Yes' if special_event else 'No'}
        
        ## Summary Metrics
        - Next 24h Traffic: {int(next_24h_traffic):,} vehicles
        - Next 24h Revenue: ${next_24h_revenue:,.0f}
        - Peak Traffic Hour: {peak_hour['timestamp'].strftime('%m/%d %H:00')}
        - Average Hourly Revenue: ${avg_hourly_revenue:.0f}
        
        ## Capacity Analysis
        - High Utilization Hours (>80%): {over_capacity_hours}
        - Peak Capacity Utilization: {max_utilization:.1f}%
        
        ## Key Insights
        {chr(10).join(['- ' + rec.replace('🔴', '').replace('💰', '').replace('📊', '').replace('🌦️', '').replace('🎉', '').replace('📈', '').replace('📉', '').replace('✅', '') for rec in recommendations])}
        
        ## Hourly Forecast (Next 24 Hours)
        {detail_predictions.head(24)[['timestamp_formatted', 'predicted_traffic_adjusted', 'predicted_revenue_adjusted']].to_string(index=False)}
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
