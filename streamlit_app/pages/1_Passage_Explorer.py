import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Passage Explorer", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle Passage Explorer")
st.markdown("Analyze vehicle passages, IVDC logs, and traffic patterns")

# Generate comprehensive sample data
@st.cache_data
def load_passage_data():
    np.random.seed(42)
    
    # Generate 30 days of hourly data
    dates = pd.date_range(end=datetime.now(), periods=720, freq='H')
    lanes = ['Lane_1', 'Lane_2', 'Lane_3', 'Lane_4', 'Lane_5']
    vehicle_types = ['Car', 'Truck', 'Bus', 'Motorcycle', 'Trailer']
    
    passages = []
    
    for date in dates:
        hour = date.hour
        weekday = date.weekday()  # 0=Monday, 6=Sunday
        
        # Traffic patterns based on time and day
        base_multiplier = 1.0
        if weekday < 5:  # Weekdays
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                base_multiplier = 3.0
            elif 10 <= hour <= 16:  # Daytime
                base_multiplier = 1.5
            elif 22 <= hour or hour <= 5:  # Night
                base_multiplier = 0.3
        else:  # Weekends
            if 10 <= hour <= 18:  # Weekend daytime
                base_multiplier = 1.8
            elif 19 <= hour <= 22:  # Weekend evening
                base_multiplier = 2.2
            else:
                base_multiplier = 0.4
        
        for lane in lanes:
            # Each lane has different characteristics
            lane_factor = {
                'Lane_1': 1.2,  # Express lane
                'Lane_2': 1.0,  # Regular
                'Lane_3': 1.0,  # Regular
                'Lane_4': 0.8,  # Truck lane
                'Lane_5': 0.6   # Emergency/special
            }
            
            base_count = int(50 * base_multiplier * lane_factor[lane])
            vehicle_count = max(0, np.random.poisson(base_count))
            
            # Generate individual vehicle records
            for i in range(vehicle_count):
                # Vehicle type probabilities vary by lane and time
                if lane == 'Lane_4':  # Truck lane
                    type_probs = [0.3, 0.5, 0.1, 0.05, 0.05]
                else:
                    type_probs = [0.75, 0.15, 0.05, 0.04, 0.01]
                
                v_type = np.random.choice(vehicle_types, p=type_probs)
                
                # Speed varies by vehicle type and conditions
                speed_ranges = {
                    'Car': (55, 75),
                    'Truck': (45, 65),
                    'Bus': (50, 65),
                    'Motorcycle': (60, 80),
                    'Trailer': (40, 60)
                }
                
                min_speed, max_speed = speed_ranges[v_type]
                speed = np.random.uniform(min_speed, max_speed)
                
                # Add congestion effects during rush hours
                if base_multiplier > 2.0:  # Heavy traffic
                    speed *= np.random.uniform(0.6, 0.8)  # Slower speeds
                
                # IVDC data (In-Vehicle Device Communication)
                ivdc_success = np.random.random() > 0.05  # 95% success rate
                
                # Generate some realistic vehicle IDs
                vehicle_id = f"{v_type[:1]}{np.random.randint(1000, 9999)}"
                
                passage_time = date + timedelta(minutes=np.random.randint(0, 60))
                
                passages.append({
                    'timestamp': passage_time,
                    'lane_id': lane,
                    'vehicle_id': vehicle_id,
                    'vehicle_type': v_type,
                    'speed_mph': round(speed, 1),
                    'length_ft': get_vehicle_length(v_type),
                    'weight_lbs': get_vehicle_weight(v_type),
                    'ivdc_success': ivdc_success,
                    'etc_tag_id': f"ETC{np.random.randint(100000, 999999)}" if np.random.random() > 0.3 else None,
                    'image_quality': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], p=[0.6, 0.25, 0.1, 0.05]),
                    'weather_condition': get_weather_condition(date),
                    'violation': np.random.random() < 0.02,  # 2% violation rate
                    'processing_time_ms': np.random.normal(150, 30)
                })
    
    return pd.DataFrame(passages)

def get_vehicle_length(vehicle_type):
    lengths = {
        'Car': np.random.normal(15, 2),
        'Truck': np.random.normal(65, 10),
        'Bus': np.random.normal(40, 5),
        'Motorcycle': np.random.normal(7, 1),
        'Trailer': np.random.normal(75, 15)
    }
    return max(5, round(lengths[vehicle_type], 1))

def get_vehicle_weight(vehicle_type):
    weights = {
        'Car': np.random.normal(3500, 500),
        'Truck': np.random.normal(35000, 8000),
        'Bus': np.random.normal(25000, 3000),
        'Motorcycle': np.random.normal(500, 100),
        'Trailer': np.random.normal(45000, 12000)
    }
    return max(200, round(weights[vehicle_type], 0))

def get_weather_condition(date):
    # Simulate seasonal weather patterns
    month = date.month
    if month in [12, 1, 2]:  # Winter
        return np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], p=[0.4, 0.3, 0.2, 0.1])
    elif month in [6, 7, 8]:  # Summer
        return np.random.choice(['Clear', 'Rain', 'Cloudy'], p=[0.7, 0.2, 0.1])
    else:  # Spring/Fall
        return np.random.choice(['Clear', 'Rain', 'Cloudy', 'Fog'], p=[0.5, 0.3, 0.15, 0.05])

# Load data
passages_df = load_passage_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range
min_date = passages_df['timestamp'].min().date()
max_date = passages_df['timestamp'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - timedelta(days=7), max_date),
    min_value=min_date,
    max_value=max_date
)

# Lane selection
selected_lanes = st.sidebar.multiselect(
    "Select Lanes",
    options=passages_df['lane_id'].unique(),
    default=passages_df['lane_id'].unique()
)

# Vehicle type selection
selected_types = st.sidebar.multiselect(
    "Select Vehicle Types",
    options=passages_df['vehicle_type'].unique(),
    default=passages_df['vehicle_type'].unique()
)

# Speed filter
min_speed, max_speed = st.sidebar.slider(
    "Speed Range (mph)",
    min_value=0.0,
    max_value=100.0,
    value=(0.0, 100.0),
    step=5.0
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = passages_df[
        (passages_df['timestamp'].dt.date >= start_date) &
        (passages_df['timestamp'].dt.date <= end_date) &
        (passages_df['lane_id'].isin(selected_lanes)) &
        (passages_df['vehicle_type'].isin(selected_types)) &
        (passages_df['speed_mph'] >= min_speed) &
        (passages_df['speed_mph'] <= max_speed)
    ]
else:
    filtered_df = passages_df

# Key metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_passages = len(filtered_df)
    st.metric("Total Passages", f"{total_passages:,}")

with col2:
    avg_speed = filtered_df['speed_mph'].mean()
    st.metric("Average Speed", f"{avg_speed:.1f} mph")

with col3:
    ivdc_success_rate = (filtered_df['ivdc_success'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("IVDC Success Rate", f"{ivdc_success_rate:.1f}%")

with col4:
    violation_rate = (filtered_df['violation'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Violation Rate", f"{violation_rate:.2f}%")

# Traffic patterns
st.subheader("🚦 Traffic Patterns")

tab1, tab2, tab3 = st.tabs(["Hourly Traffic", "Daily Patterns", "Lane Distribution"])

with tab1:
    hourly_traffic = filtered_df.groupby(filtered_df['timestamp'].dt.hour).size().reset_index(name='count')
    fig = px.line(hourly_traffic, x='timestamp', y='count', 
                  title="Traffic Volume by Hour of Day",
                  labels={'timestamp': 'Hour of Day', 'count': 'Vehicle Count'})
    fig.update_layout(xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    daily_traffic = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reset_index(name='count')
    fig = px.bar(daily_traffic, x='timestamp', y='count',
                 title="Daily Traffic Volume",
                 labels={'timestamp': 'Date', 'count': 'Vehicle Count'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    lane_distribution = filtered_df['lane_id'].value_counts().reset_index()
    lane_distribution.columns = ['lane_id', 'count']
    fig = px.bar(lane_distribution, x='lane_id', y='count',
                 title="Traffic Distribution by Lane",
                 labels={'lane_id': 'Lane', 'count': 'Vehicle Count'})
    st.plotly_chart(fig, use_container_width=True)

# Vehicle analysis
st.subheader("🚗 Vehicle Analysis")

col1, col2 = st.columns(2)

with col1:
    # Vehicle type distribution
    type_dist = filtered_df['vehicle_type'].value_counts()
    fig = px.pie(values=type_dist.values, names=type_dist.index,
                 title="Vehicle Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Speed distribution by vehicle type
    fig = px.box(filtered_df, x='vehicle_type', y='speed_mph',
                 title="Speed Distribution by Vehicle Type")
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# IVDC Analysis
st.subheader("📡 IVDC (In-Vehicle Device Communication) Analysis")

col1, col2 = st.columns(2)

with col1:
    # IVDC success rate by lane
    ivdc_by_lane = filtered_df.groupby('lane_id')['ivdc_success'].mean().reset_index()
    ivdc_by_lane['success_rate'] = ivdc_by_lane['ivdc_success'] * 100
    
    fig = px.bar(ivdc_by_lane, x='lane_id', y='success_rate',
                 title="IVDC Success Rate by Lane",
                 labels={'lane_id': 'Lane', 'success_rate': 'Success Rate (%)'})
    fig.update_layout(yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Processing time distribution
    fig = px.histogram(filtered_df, x='processing_time_ms',
                      title="Processing Time Distribution",
                      labels={'processing_time_ms': 'Processing Time (ms)', 'count': 'Frequency'})
    st.plotly_chart(fig, use_container_width=True)

# Weather impact
st.subheader("🌦️ Weather Impact Analysis")

weather_analysis = filtered_df.groupby('weather_condition').agg({
    'speed_mph': 'mean',
    'processing_time_ms': 'mean',
    'ivdc_success': 'mean'
}).reset_index()

weather_analysis['ivdc_success'] *= 100  # Convert to percentage

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(weather_analysis, x='weather_condition', y='speed_mph',
                 title="Average Speed by Weather Condition",
                 labels={'weather_condition': 'Weather', 'speed_mph': 'Average Speed (mph)'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(weather_analysis, x='weather_condition', y='ivdc_success',
                 title="IVDC Success Rate by Weather",
                 labels={'weather_condition': 'Weather', 'ivdc_success': 'Success Rate (%)'})
    st.plotly_chart(fig, use_container_width=True)

# Data quality metrics
st.subheader("📋 Data Quality Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    image_quality = filtered_df['image_quality'].value_counts()
    fig = px.pie(values=image_quality.values, names=image_quality.index,
                 title="Image Quality Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    etc_coverage = (filtered_df['etc_tag_id'].notna().sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("ETC Tag Coverage", f"{etc_coverage:.1f}%")
    
    processing_avg = filtered_df['processing_time_ms'].mean()
    st.metric("Avg Processing Time", f"{processing_avg:.0f} ms")

with col3:
    violations = filtered_df[filtered_df['violation'] == True]
    if not violations.empty:
        violation_by_type = violations['vehicle_type'].value_counts()
        fig = px.bar(x=violation_by_type.index, y=violation_by_type.values,
                     title="Violations by Vehicle Type",
                     labels={'x': 'Vehicle Type', 'y': 'Violation Count'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No violations in selected time period")

# Raw data table
st.subheader("📄 Recent Passage Records")
st.dataframe(
    filtered_df.head(100)[['timestamp', 'lane_id', 'vehicle_type', 'speed_mph', 
                          'ivdc_success', 'weather_condition', 'image_quality']],
    use_container_width=True
)

# Export options
st.subheader("💾 Export Data")
col1, col2 = st.columns(2)

with col1:
    if st.button("Export Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"passage_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Generate Summary Report"):
        summary = f"""
        # Passage Data Summary Report
        
        **Time Period**: {date_range[0] if len(date_range) == 2 else 'All Time'} to {date_range[1] if len(date_range) == 2 else 'All Time'}
        **Total Passages**: {total_passages:,}
        **Average Speed**: {avg_speed:.1f} mph
        **IVDC Success Rate**: {ivdc_success_rate:.1f}%
        **Violation Rate**: {violation_rate:.2f}%
        
        **Lanes Analyzed**: {', '.join(selected_lanes)}
        **Vehicle Types**: {', '.join(selected_types)}
        """
        
        st.download_button(
            label="Download Report",
            data=summary,
            file_name=f"passage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
