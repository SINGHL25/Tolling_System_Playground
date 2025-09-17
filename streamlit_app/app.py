
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Tolling System Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample data for the dashboard"""
    np.random.seed(42)
    
    # Generate dates for the last 30 days
    dates = pd.date_range(end=datetime.now(), periods=720, freq='H')
    
    # Passage data
    passage_data = []
    for date in dates:
        hour = date.hour
        # More traffic during rush hours (7-9 AM, 5-7 PM)
        base_traffic = 50
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_traffic = 150
        elif 10 <= hour <= 16:
            base_traffic = 80
        elif 22 <= hour or hour <= 5:
            base_traffic = 20
        
        vehicles = np.random.poisson(base_traffic)
        passage_data.append({
            'timestamp': date,
            'lane_id': np.random.choice(['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4']),
            'vehicle_count': vehicles,
            'avg_speed': np.random.normal(65, 10),
            'vehicle_type': np.random.choice(['Car', 'Truck', 'Bus', 'Motorcycle'], 
                                           p=[0.7, 0.2, 0.05, 0.05])
        })
    
    passages_df = pd.DataFrame(passage_data)
    
    # Transaction data
    transaction_data = []
    toll_rates = {'Car': 5.0, 'Truck': 15.0, 'Bus': 8.0, 'Motorcycle': 2.5}
    
    for _, row in passages_df.iterrows():
        for _ in range(row['vehicle_count']):
            vehicle_type = row['vehicle_type']
            base_toll = toll_rates[vehicle_type]
            # Add some variation
            toll_amount = base_toll + np.random.normal(0, 0.5)
            toll_amount = max(toll_amount, base_toll * 0.8)  # Minimum 80% of base rate
            
            transaction_data.append({
                'timestamp': row['timestamp'],
                'lane_id': row['lane_id'],
                'vehicle_type': vehicle_type,
                'toll_amount': toll_amount,
                'payment_method': np.random.choice(['ETC', 'Cash', 'Card'], p=[0.6, 0.25, 0.15])
            })
    
    transactions_df = pd.DataFrame(transaction_data)
    
    # System health data
    health_data = []
    systems = ['Camera_1', 'Camera_2', 'Sensor_A', 'Sensor_B', 'Payment_Terminal_1', 'Payment_Terminal_2']
    
    for date in dates:
        for system in systems:
            uptime = np.random.uniform(95, 99.8)  # High uptime
            health_data.append({
                'timestamp': date,
                'system_name': system,
                'uptime_percent': uptime,
                'status': 'Online' if uptime > 98 else 'Degraded' if uptime > 95 else 'Offline',
                'temperature': np.random.normal(35, 5),
                'error_count': np.random.poisson(0.1)
            })
    
    health_df = pd.DataFrame(health_data)
    
    # Pollution data
    pollution_data = []
    for date in dates:
        # Higher pollution during rush hours
        hour = date.hour
        base_pm25 = 25
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_pm25 = 45
        
        pollution_data.append({
            'timestamp': date,
            'pm25': np.random.normal(base_pm25, 8),
            'pm10': np.random.normal(base_pm25 * 1.5, 12),
            'no2': np.random.normal(40, 10),
            'co': np.random.normal(1.2, 0.3)
        })
    
    pollution_df = pd.DataFrame(pollution_data)
    
    return passages_df, transactions_df, health_df, pollution_df

# Load sample data
@st.cache_data
def load_data():
    return generate_sample_data()

passages_df, transactions_df, health_df, pollution_df = load_data()

# Sidebar
st.sidebar.image("https://via.placeholder.com/200x100/1f77b4/white?text=Toll+System", width=200)
st.sidebar.title("🚗 Toll Dashboard")
st.sidebar.markdown("---")

# Date range selector
today = datetime.now().date()
start_date = st.sidebar.date_input("Start Date", today - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", today)

# Main dashboard
st.markdown('<h1 class="main-header">🚗 Tolling System Dashboard</h1>', unsafe_allow_html=True)

# Filter data by date range
mask = (passages_df['timestamp'].dt.date >= start_date) & (passages_df['timestamp'].dt.date <= end_date)
filtered_passages = passages_df[mask]
filtered_transactions = transactions_df[transactions_df['timestamp'].dt.date >= start_date]
filtered_transactions = filtered_transactions[filtered_transactions['timestamp'].dt.date <= end_date]

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_vehicles = filtered_passages['vehicle_count'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-value">{total_vehicles:,}</div>
        <div class="kpi-label">Total Vehicles</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_revenue = filtered_transactions['toll_amount'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-value">${total_revenue:,.0f}</div>
        <div class="kpi-label">Total Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_speed = filtered_passages['avg_speed'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-value">{avg_speed:.1f} mph</div>
        <div class="kpi-label">Average Speed</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    system_uptime = health_df[health_df['timestamp'].dt.date >= start_date]['uptime_percent'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="kpi-value">{system_uptime:.1f}%</div>
        <div class="kpi-label">System Uptime</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Traffic Flow by Hour")
    hourly_traffic = filtered_passages.groupby(filtered_passages['timestamp'].dt.hour)['vehicle_count'].sum().reset_index()
    fig = px.bar(hourly_traffic, x='timestamp', y='vehicle_count', 
                 title="Vehicle Count by Hour of Day",
                 labels={'timestamp': 'Hour', 'vehicle_count': 'Vehicle Count'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Revenue by Vehicle Type")
    revenue_by_type = filtered_transactions.groupby('vehicle_type')['toll_amount'].sum().reset_index()
    fig = px.pie(revenue_by_type, values='toll_amount', names='vehicle_type', 
                 title="Revenue Distribution by Vehicle Type")
    st.plotly_chart(fig, use_container_width=True)

# Daily trends
st.subheader("📈 Daily Trends")
daily_stats = filtered_passages.groupby(filtered_passages['timestamp'].dt.date).agg({
    'vehicle_count': 'sum',
    'avg_speed': 'mean'
}).reset_index()
daily_revenue = filtered_transactions.groupby(filtered_transactions['timestamp'].dt.date)['toll_amount'].sum().reset_index()
daily_stats = daily_stats.merge(daily_revenue, left_on='timestamp', right_on='timestamp', how='left')

fig = go.Figure()
fig.add_trace(go.Scatter(x=daily_stats['timestamp'], y=daily_stats['vehicle_count'],
                        mode='lines+markers', name='Vehicle Count', yaxis='y'))
fig.add_trace(go.Scatter(x=daily_stats['timestamp'], y=daily_stats['toll_amount'],
                        mode='lines+markers', name='Revenue ($)', yaxis='y2'))

fig.update_layout(
    title="Daily Traffic and Revenue Trends",
    xaxis_title="Date",
    yaxis=dict(title="Vehicle Count", side="left"),
    yaxis2=dict(title="Revenue ($)", side="right", overlaying="y"),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Recent data tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Recent System Alerts")
    recent_health = health_df[health_df['status'] != 'Online'].tail(10)
    if not recent_health.empty:
        st.dataframe(recent_health[['timestamp', 'system_name', 'status', 'uptime_percent']], 
                    use_container_width=True)
    else:
        st.success("All systems operating normally!")

with col2:
    st.subheader("🌍 Current Air Quality")
    latest_pollution = pollution_df.tail(1)
    if not latest_pollution.empty:
        pm25 = latest_pollution['pm25'].iloc[0]
        pm10 = latest_pollution['pm10'].iloc[0]
        no2 = latest_pollution['no2'].iloc[0]
        
        quality = "Good" if pm25 < 35 else "Moderate" if pm25 < 55 else "Poor"
        color = "green" if quality == "Good" else "orange" if quality == "Moderate" else "red"
        
        st.markdown(f"""
        **PM2.5:** {pm25:.1f} μg/m³  
        **PM10:** {pm10:.1f} μg/m³  
        **NO2:** {no2:.1f} μg/m³  
        **Air Quality:** <span style="color: {color}; font-weight: bold;">{quality}</span>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚗 Tolling System Dashboard | Last updated: {} | 
    <a href="#" onclick="window.location.reload()">Refresh Data</a>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
