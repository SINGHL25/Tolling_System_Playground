import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotly.figure_factory as ff

st.set_page_config(page_title="Congestion Monitor", page_icon="🚦", layout="wide")

st.title("🚦 Traffic Congestion Monitor")
st.markdown("Real-time traffic flow analysis, congestion KPIs, and traffic management insights")

# Generate comprehensive congestion data
@st.cache_data
def load_congestion_data():
    np.random.seed(42)
    
    # Generate 30 days of data with 15-minute intervals
    dates = pd.date_range(end=datetime.now(), periods=2880, freq='15T')
    lanes = ['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4', 'Lane_5']
    
    congestion_data = []
    
    for date in dates:
        hour = date.hour
        weekday = date.weekday()
        minute = date.minute
        
        # Base traffic patterns
        base_volume = 20  # vehicles per 15-minute interval
        
        if weekday < 5:  # Weekdays
            if 7 <= hour <= 9:  # Morning rush
                volume_multiplier = 4.0
                congestion_level = "High"
            elif 17 <= hour <= 19:  # Evening rush
                volume_multiplier = 4.5
                congestion_level = "High"
            elif 12 <= hour <= 14:  # Lunch time
                volume_multiplier = 2.0
                congestion_level = "Medium"
            elif 10 <= hour <= 16:  # Daytime
                volume_multiplier = 1.5
                congestion_level = "Low"
            elif 20 <= hour <= 22:  # Evening
                volume_multiplier = 1.8
                congestion_level = "Medium"
            else:  # Night/early morning
                volume_multiplier = 0.3
                congestion_level = "Free Flow"
        else:  # Weekends
            if 10 <= hour <= 18:  # Weekend peak
                volume_multiplier = 2.5
                congestion_level = "Medium"
            elif 19 <= hour <= 22:  # Weekend evening
                volume_multiplier = 3.0
                congestion_level = "High"
            else:
                volume_multiplier = 0.5
                congestion_level = "Low"
        
        for lane in lanes:
            # Lane-specific factors
            lane_factors = {
                'Lane_1': 1.3,  # Express lane - higher speeds, less congestion
                'Lane_2': 1.0,  # Regular lane
                'Lane_3': 1.0,  # Regular lane  
                'Lane_4': 0.7,  # Truck lane - slower, more congested
                'Lane_5': 0.4   # Emergency/service lane
            }
            
            volume = max(0, np.random.poisson(base_volume * volume_multiplier * lane_factors[lane]))
            
            # Calculate congestion metrics based on volume
            capacity = 80  # Max vehicles per 15-min interval
            occupancy = min(100, (volume / capacity) * 100)
            
            # Speed calculation based on congestion
            free_flow_speed = 65
            if occupancy < 30:
                avg_speed = np.random.normal(free_flow_speed, 5)
                actual_congestion = "Free Flow"
            elif occupancy < 60:
                avg_speed = np.random.normal(free_flow_speed * 0.8, 8)
                actual_congestion = "Light"
            elif occupancy < 80:
                avg_speed = np.random.normal(free_flow_speed * 0.6, 10)
                actual_congestion = "Moderate"
            else:
                avg_speed = np.random.normal(free_flow_speed * 0.4, 5)
                actual_congestion = "Heavy"
            
            avg_speed = max(10, avg_speed)  # Minimum speed
            
            # Wait times and delays
            if actual_congestion == "Free Flow":
                avg_wait_time = np.random.uniform(0, 2)
                delay_minutes = 0
            elif actual_congestion == "Light":
                avg_wait_time = np.random.uniform(1, 5)
                delay_minutes = np.random.uniform(0, 2)
            elif actual_congestion == "Moderate":
                avg_wait_time = np.random.uniform(3, 12)
                delay_minutes = np.random.uniform(2, 8)
            else:  # Heavy
                avg_wait_time = np.random.uniform(8, 25)
                delay_minutes = np.random.uniform(5, 20)
            
            # Travel time index (normal travel time = 1.0)
            travel_time_index = 1.0 + (delay_minutes / 30)  # 30 min normal travel time
            
            # Queue length estimation
            if actual_congestion == "Free Flow":
                queue_length = 0
            else:
                queue_length = max(0, np.random.poisson(volume * 0.3))
            
            # Environmental impact
            fuel_efficiency_loss = max(0, (100 - occupancy) / 100 * 0.3)  # Up to 30% loss
            co2_emissions = volume * (2.3 + fuel_efficiency_loss)  # kg CO2 per vehicle
            
            congestion_data.append({
                'timestamp': date,
                'lane_id': lane,
                'vehicle_count': volume,
                'lane_capacity': capacity,
                'occupancy_percent': round(occupancy, 1),
                'avg_speed_mph': round(avg_speed, 1),
                'free_flow_speed': free_flow_speed,
                'congestion_level': actual_congestion,
                'avg_wait_time_sec': round(avg_wait_time, 1),
                'delay_minutes': round(delay_minutes, 1),
                'travel_time_index': round(travel_time_index, 2),
                'queue_length_vehicles': queue_length,
                'throughput_vph': volume * 4,  # vehicles per hour
                'density_vpm': volume / 2 if avg_speed > 0 else 0,  # vehicles per mile
                'co2_emissions_kg': round(co2_emissions, 1),
                'fuel_efficiency_loss_percent': round(fuel_efficiency_loss * 100, 1),
                'incident_reported': np.random.random() < 0.02,  # 2% chance of incident
                'weather_impact': np.random.choice(['None', 'Light Rain', 'Heavy Rain', 'Fog'], 
                                                 p=[0.7, 0.2, 0.05, 0.05])
            })
    
    return pd.DataFrame(congestion_data)

# Load congestion data
congestion_df = load_congestion_data()

# Sidebar filters
st.sidebar.header("🚦 Traffic Filters")

# Date range
min_date = congestion_df['timestamp'].min().date()
max_date = congestion_df['timestamp'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - timedelta(days=3), max_date),
    min_value=min_date,
    max_value=max_date
)

# Time of day filter
time_filter = st.sidebar.selectbox(
    "Time Period",
    options=['All Day', 'Morning Rush (7-9 AM)', 'Evening Rush (5-7 PM)', 'Off-Peak', 'Night (10 PM - 6 AM)'],
    index=0
)

# Lane selection
selected_lanes = st.sidebar.multiselect(
    "Select Lanes",
    options=congestion_df['lane_id'].unique(),
    default=congestion_df['lane_id'].unique()
)

# Congestion level filter
congestion_filter = st.sidebar.multiselect(
    "Congestion Levels",
    options=['Free Flow', 'Light', 'Moderate', 'Heavy'],
    default=['Free Flow', 'Light', 'Moderate', 'Heavy']
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = congestion_df[
        (congestion_df['timestamp'].dt.date >= start_date) &
        (congestion_df['timestamp'].dt.date <= end_date) &
        (congestion_df['lane_id'].isin(selected_lanes)) &
        (congestion_df['congestion_level'].isin(congestion_filter))
    ]
else:
    filtered_df = congestion_df

# Apply time filter
if time_filter == 'Morning Rush (7-9 AM)':
    filtered_df = filtered_df[filtered_df['timestamp'].dt.hour.between(7, 9)]
elif time_filter == 'Evening Rush (5-7 PM)':
    filtered_df = filtered_df[filtered_df['timestamp'].dt.hour.between(17, 19)]
elif time_filter == 'Off-Peak':
    filtered_df = filtered_df[
        ~filtered_df['timestamp'].dt.hour.between(7, 9) & 
        ~filtered_df['timestamp'].dt.hour.between(17, 19)
    ]
elif time_filter == 'Night (10 PM - 6 AM)':
    filtered_df = filtered_df[
        (filtered_df['timestamp'].dt.hour >= 22) | 
        (filtered_df['timestamp'].dt.hour <= 6)
    ]

# Real-time status (latest data point)
st.subheader("🔴 Current Traffic Status")
latest_data = congestion_df.tail(len(congestion_df['lane_id'].unique()))

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_occupancy = latest_data['occupancy_percent'].mean()
    occupancy_status = "🟢 Normal" if avg_occupancy < 50 else "🟡 Busy" if avg_occupancy < 75 else "🔴 Congested"
    st.metric("Average Occupancy", f"{avg_occupancy:.1f}%", delta=occupancy_status)

with col2:
    avg_speed = latest_data['avg_speed_mph'].mean()
    speed_status = "Fast" if avg_speed > 55 else "Slow" if avg_speed < 35 else "Normal"
    st.metric("Average Speed", f"{avg_speed:.1f} mph", delta=speed_status)

with col3:
    avg_delay = latest_data['delay_minutes'].mean()
    st.metric("Average Delay", f"{avg_delay:.1f} min")

with col4:
    total_throughput = latest_data['throughput_vph'].sum()
    st.metric("Total Throughput", f"{total_throughput:,} vph")

with col5:
    incidents = latest_data['incident_reported'].sum()
    incident_status = "🟢 Clear" if incidents == 0 else f"🔴 {incidents} Active"
    st.metric("Incidents", incident_status)

# Traffic heatmap
st.subheader("🗺️ Traffic Flow Heatmap")

# Create hourly heatmap data
heatmap_data = filtered_df.groupby([
    filtered_df['timestamp'].dt.hour,
    'lane_id'
])['occupancy_percent'].mean().unstack(fill_value=0)

if not heatmap_data.empty:
    fig = px.imshow(
        heatmap_data.T,
        labels=dict(x="Hour of Day", y="Lane", color="Occupancy %"),
        title="Traffic Occupancy Heatmap by Hour and Lane",
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Key Performance Indicators
st.subheader("📊 Congestion KPIs")

col1, col2 = st.columns(2)

with col1:
    # Congestion level distribution
    congestion_dist = filtered_df['congestion_level'].value_counts()
    colors = {'Free Flow': '#2E8B57', 'Light': '#FFD700', 'Moderate': '#FF8C00', 'Heavy': '#DC143C'}
    fig = px.pie(
        values=congestion_dist.values,
        names=congestion_dist.index,
        title="Congestion Level Distribution",
        color=congestion_dist.index,
        color_discrete_map=colors
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Average speed by congestion level
    speed_by_congestion = filtered_df.groupby('congestion_level')['avg_speed_mph'].mean().reset_index()
    fig = px.bar(
        speed_by_congestion,
        x='congestion_level',
        y='avg_speed_mph',
        title="Average Speed by Congestion Level",
        labels={'avg_speed_mph': 'Speed (mph)', 'congestion_level': 'Congestion Level'},
        color='avg_speed_mph',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

# Time series analysis
st.subheader("📈 Traffic Flow Trends")

tab1, tab2, tab3 = st.tabs(["Occupancy Trends", "Speed Analysis", "Wait Times"])

with tab1:
    # Occupancy over time
    hourly_occupancy = filtered_df.groupby([
        filtered_df['timestamp'].dt.floor('H'),
        'lane_id'
    ])['occupancy_percent'].mean().reset_index()
    
    fig = px.line(
        hourly_occupancy,
        x='timestamp',
        y='occupancy_percent',
        color='lane_id',
        title="Lane Occupancy Over Time",
        labels={'occupancy_percent': 'Occupancy (%)', 'timestamp': 'Time'}
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="Critical Threshold (80%)")
    fig.add_hline(y=60, line_dash="dash", line_color="orange",
                  annotation_text="Warning Threshold (60%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Speed analysis
    hourly_speed = filtered_df.groupby([
        filtered_df['timestamp'].dt.floor('H'),
        'lane_id'
    ])['avg_speed_mph'].mean().reset_index()
    
    fig = px.line(
        hourly_speed,
        x='timestamp',
        y='avg_speed_mph',
        color='lane_id',
        title="Average Speed Trends",
        labels={'avg_speed_mph': 'Speed (mph)', 'timestamp': 'Time'}
    )
    fig.add_hline(y=65, line_dash="dash", line_color="green",
                  annotation_text="Free Flow Speed")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Wait times
    hourly_wait = filtered_df.groupby([
        filtered_df['timestamp'].dt.floor('H'),
        'lane_id'
    ])['avg_wait_time_sec'].mean().reset_index()
    
    fig = px.line(
        hourly_wait,
        x='timestamp',
        y='avg_wait_time_sec',
        color='lane_id',
        title="Average Wait Times",
        labels={'avg_wait_time_sec': 'Wait Time (seconds)', 'timestamp': 'Time'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Congestion patterns analysis
st.subheader("🔄 Traffic Pattern Analysis")

col1, col2 = st.columns(2)

with col1:
    # Weekly patterns
    filtered_df['weekday'] = filtered_df['timestamp'].dt.day_name()
    weekly_pattern = filtered_df.groupby('weekday')['occupancy_percent'].mean().reset_index()
    
    # Reorder weekdays
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pattern['weekday'] = pd.Categorical(weekly_pattern['weekday'], categories=weekday_order, ordered=True)
    weekly_pattern = weekly_pattern.sort_values('weekday')
    
    fig = px.bar(
        weekly_pattern,
        x='weekday',
        y='occupancy_percent',
        title="Average Occupancy by Day of Week",
        labels={'occupancy_percent': 'Occupancy (%)', 'weekday': 'Day of Week'}
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Hourly patterns
    hourly_pattern = filtered_df.groupby(filtered_df['timestamp'].dt.hour)['occupancy_percent'].mean().reset_index()
    
    fig = px.line(
        hourly_pattern,
        x='timestamp',
        y='occupancy_percent',
        title="Average Occupancy by Hour of Day",
        labels={'timestamp': 'Hour', 'occupancy_percent': 'Occupancy (%)'},
        markers=True
    )
    fig.update_layout(xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig, use_container_width=True)

# Lane performance comparison
st.subheader("🛣️ Lane Performance Comparison")

lane_performance = filtered_df.groupby('lane_id').agg({
    'occupancy_percent': 'mean',
    'avg_speed_mph': 'mean',
    'throughput_vph': 'mean',
    'delay_minutes': 'mean',
    'queue_length_vehicles': 'mean',
    'incident_reported': 'sum'
}).reset_index()

col1, col2 = st.columns(2)

with col1:
    # Throughput vs Occupancy
    fig = px.scatter(
        lane_performance,
        x='occupancy_percent',
        y='throughput_vph',
        size='avg_speed_mph',
        color='lane_id',
        title="Throughput vs Occupancy by Lane",
        labels={
            'occupancy_percent': 'Average Occupancy (%)',
            'throughput_vph': 'Throughput (vehicles/hour)'
        },
        hover_data=['delay_minutes']
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Lane efficiency metrics
    lane_performance['efficiency_score'] = (
        lane_performance['throughput_vph'] / 
        (lane_performance['delay_minutes'] + 1)  # Add 1 to avoid division by zero
    )
    
    fig = px.bar(
        lane_performance.sort_values('efficiency_score', ascending=True),
        y='lane_id',
        x='efficiency_score',
        title="Lane Efficiency Score",
        labels={'efficiency_score': 'Efficiency Score', 'lane_id': 'Lane'},
        orientation='h'
    )
    st.plotly_chart(fig, use_container_width=True)

# Environmental impact
st.subheader("🌱 Environmental Impact")

col1, col2, col3 = st.columns(3)

with col1:
    total_co2 = filtered_df['co2_emissions_kg'].sum()
    st.metric("Total CO2 Emissions", f"{total_co2:,.0f} kg")

with col2:
    avg_fuel_loss = filtered_df['fuel_efficiency_loss_percent'].mean()
    st.metric("Avg Fuel Efficiency Loss", f"{avg_fuel_loss:.1f}%")

with col3:
    # Calculate environmental cost of congestion
    congested_data = filtered_df[filtered_df['congestion_level'].isin(['Moderate', 'Heavy'])]
    environmental_cost = congested_data['co2_emissions_kg'].sum() * 0.05  # $0.05 per kg CO2
    st.metric("Congestion Environmental Cost", f"${environmental_cost:.2f}")

# CO2 emissions by congestion level
emissions_by_congestion = filtered_df.groupby('congestion_level')['co2_emissions_kg'].sum().reset_index()
fig = px.bar(
    emissions_by_congestion,
    x='congestion_level',
    y='co2_emissions_kg',
    title="CO2 Emissions by Congestion Level",
    labels={'co2_emissions_kg': 'CO2 Emissions (kg)', 'congestion_level': 'Congestion Level'},
    color='co2_emissions_kg',
    color_continuous_scale='Reds'
)
st.plotly_chart(fig, use_container_width=True)

# Incident analysis
st.subheader("🚨 Incident Impact Analysis")

if filtered_df['incident_reported'].sum() > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Incidents by lane
        incidents_by_lane = filtered_df.groupby('lane_id')['incident_reported'].sum().reset_index()
        fig = px.bar(
            incidents_by_lane,
            x='lane_id',
            y='incident_reported',
            title="Incidents by Lane",
            labels={'incident_reported': 'Number of Incidents', 'lane_id': 'Lane'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Impact of incidents on speed
        incident_impact = filtered_df.groupby('incident_reported')['avg_speed_mph'].mean().reset_index()
        incident_impact['incident_status'] = incident_impact['incident_reported'].map({False: 'No Incident', True: 'Incident'})
        
        fig = px.bar(
            incident_impact,
            x='incident_status',
            y='avg_speed_mph',
            title="Speed Impact of Incidents",
            labels={'avg_speed_mph': 'Average Speed (mph)', 'incident_status': 'Incident Status'}
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No incidents reported in selected time period")

# Traffic management recommendations
st.subheader("🎯 Traffic Management Recommendations")

# Analyze current conditions and provide recommendations
high_congestion_lanes = lane_performance[lane_performance['occupancy_percent'] > 70]
slow_lanes = lane_performance[lane_performance['avg_speed_mph'] < 45]
high_delay_lanes = lane_performance[lane_performance['delay_minutes'] > 5]

recommendations = []

if not high_congestion_lanes.empty:
    recommendations.append(f"🔴 **High Congestion Alert**: Lanes {', '.join(high_congestion_lanes['lane_id'])} are experiencing high occupancy (>70%). Consider implementing dynamic pricing or ramp metering.")

if not slow_lanes.empty:
    recommendations.append(f"🟡 **Speed Alert**: Lanes {', '.join(slow_lanes['lane_id'])} have reduced speeds (<45 mph). Check for incidents or implement speed harmonization.")

if not high_delay_lanes.empty:
    recommendations.append(f"⏰ **Delay Alert**: Lanes {', '.join(high_delay_lanes['lane_id'])} are experiencing significant delays (>5 min). Consider opening additional lanes or adjusting signal timing.")

if filtered_df['weather_impact'].value_counts().get('Heavy Rain', 0) > 0:
    recommendations.append("🌧️ **Weather Impact**: Heavy rain detected. Implement reduced speed limits and increase following distances.")

if len(recommendations) == 0:
    recommendations.append("✅ **All Clear**: Traffic conditions are operating within normal parameters.")

for rec in recommendations:
    st.markdown(rec)

# Performance dashboard
st.subheader("📋 Congestion Summary Table")

summary_stats = filtered_df.groupby('lane_id').agg({
    'occupancy_percent': ['mean', 'max'],
    'avg_speed_mph': ['mean', 'min'],
    'delay_minutes': ['mean', 'max'],
    'throughput_vph': 'mean',
    'queue_length_vehicles': 'mean',
    'incident_reported': 'sum',
    'co2_emissions_kg': 'sum'
}).round(2)

# Flatten column names
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
summary_stats = summary_stats.reset_index()

st.dataframe(summary_stats, use_container_width=True)

# Export options
st.subheader("📤 Export Congestion Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Congestion Data"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"congestion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Generate Traffic Report"):
        report = f"""
        # Traffic Congestion Report
        
        **Analysis Period**: {date_range[0]} to {date_range[1]}
        **Time Filter**: {time_filter}
        
        ## Key Metrics
        - Average Occupancy: {filtered_df['occupancy_percent'].mean():.1f}%
        - Average Speed: {filtered_df['avg_speed_mph'].mean():.1f} mph
        - Average Delay: {filtered_df['delay_minutes'].mean():.1f} minutes
        - Total Incidents: {filtered_df['incident_reported'].sum()}
        - Total CO2 Emissions: {filtered_df['co2_emissions_kg'].sum():.0f} kg
        
        ## Recommendations
        {chr(10).join(['- ' + rec.replace('🔴', '').replace('🟡', '').replace('⏰', '').replace('🌧️', '').replace('✅', '') for rec in recommendations])}
        
        ## Lane Performance Summary
        {summary_stats.to_string(index=False)}
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
