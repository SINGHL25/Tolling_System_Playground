import streamlit as st
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta, time as dt_time
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Tolling System Playground",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .passage-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9f9f9;
    }
    .alert-high {
        border-left: 5px solid #ff4444;
        background: #fff5f5;
    }
    .alert-medium {
        border-left: 5px solid #ffaa00;
        background: #fffaf0;
    }
    .alert-low {
        border-left: 5px solid #44ff44;
        background: #f0fff4;
    }
    .kpi-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
    }
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Enums and Data Classes
class VehicleClass(Enum):
    CAR = "Car"
    MOTORCYCLE = "Motorcycle"
    TRUCK = "Truck"
    BUS = "Bus"
    VAN = "Van"
    TRAILER = "Trailer"

class LaneType(Enum):
    ELECTRONIC = "Electronic"
    MANUAL = "Manual" 
    MIXED = "Mixed"
    EMERGENCY = "Emergency"

class TransactionStatus(Enum):
    SUCCESS = "Success"
    FAILED = "Failed"
    PENDING = "Pending"
    DISPUTED = "Disputed"

class SystemStatus(Enum):
    ONLINE = "Online"
    OFFLINE = "Offline"
    MAINTENANCE = "Maintenance"
    ERROR = "Error"

@dataclass
class Passage:
    id: str
    timestamp: datetime
    lane_id: str
    vehicle_class: VehicleClass
    license_plate: str
    speed: float
    length: float
    height: float
    weight: float
    tag_id: Optional[str]
    image_quality: float
    temperature: float
    
@dataclass
class Transaction:
    id: str
    passage_id: str
    timestamp: datetime
    amount: float
    status: TransactionStatus
    payment_method: str
    toll_plaza_id: str
    lane_id: str
    vehicle_class: VehicleClass
    discount_applied: float
    
@dataclass
class IVDCRecord:
    id: str
    timestamp: datetime
    lane_id: str
    validation_status: str
    error_code: Optional[str]
    confidence_score: float
    processing_time: float
    
@dataclass
class SystemHealth:
    device_id: str
    timestamp: datetime
    status: SystemStatus
    uptime_hours: float
    temperature: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_count: int

class TollingSystemSimulator:
    """Comprehensive simulation engine for tolling system operations"""
    
    def __init__(self):
        self.passages = []
        self.transactions = []
        self.ivdc_records = []
        self.system_health = []
        self.pollution_data = []
        
        # Configuration
        self.toll_rates = {
            VehicleClass.CAR: 2.50,
            VehicleClass.MOTORCYCLE: 1.25,
            VehicleClass.TRUCK: 5.00,
            VehicleClass.BUS: 4.00,
            VehicleClass.VAN: 3.00,
            VehicleClass.TRAILER: 7.50
        }
        
        self.lanes = [f"Lane_{i:02d}" for i in range(1, 13)]
        self.toll_plazas = ["Plaza_North", "Plaza_South", "Plaza_East", "Plaza_West"]
        self.payment_methods = ["ETC", "Credit_Card", "Cash", "Mobile_Pay"]
        
    def generate_passage(self) -> Passage:
        """Generate a realistic vehicle passage"""
        passage_id = f"P{random.randint(100000, 999999)}"
        timestamp = datetime.now() - timedelta(
            minutes=random.randint(0, 1440),
            seconds=random.randint(0, 3600)
        )
        
        vehicle_class = random.choice(list(VehicleClass))
        lane_id = random.choice(self.lanes)
        
        # Generate license plate
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
        numbers = ''.join(random.choices('0123456789', k=3))
        license_plate = f"{letters}{numbers}"
        
        # Vehicle characteristics based on class
        if vehicle_class == VehicleClass.CAR:
            speed = random.uniform(30, 80)
            length = random.uniform(4.0, 5.5)
            height = random.uniform(1.4, 1.8)
            weight = random.uniform(1000, 2000)
        elif vehicle_class == VehicleClass.MOTORCYCLE:
            speed = random.uniform(25, 90)
            length = random.uniform(1.8, 2.5)
            height = random.uniform(1.0, 1.3)
            weight = random.uniform(150, 400)
        elif vehicle_class == VehicleClass.TRUCK:
            speed = random.uniform(40, 70)
            length = random.uniform(8.0, 16.0)
            height = random.uniform(2.5, 4.0)
            weight = random.uniform(3000, 40000)
        elif vehicle_class == VehicleClass.BUS:
            speed = random.uniform(30, 65)
            length = random.uniform(10.0, 15.0)
            height = random.uniform(3.0, 3.8)
            weight = random.uniform(8000, 18000)
        else:  # VAN, TRAILER
            speed = random.uniform(35, 75)
            length = random.uniform(5.0, 20.0)
            height = random.uniform(2.0, 4.0)
            weight = random.uniform(2000, 35000)
        
        # Tag ID (80% have electronic tags)
        tag_id = f"TAG{random.randint(1000000, 9999999)}" if random.random() < 0.8 else None
        
        image_quality = np.random.beta(6, 2)  # Generally good quality
        temperature = random.uniform(15, 35)  # Environmental temperature
        
        return Passage(
            id=passage_id,
            timestamp=timestamp,
            lane_id=lane_id,
            vehicle_class=vehicle_class,
            license_plate=license_plate,
            speed=speed,
            length=length,
            height=height,
            weight=weight,
            tag_id=tag_id,
            image_quality=image_quality,
            temperature=temperature
        )
    
    def generate_transaction(self, passage: Passage) -> Transaction:
        """Generate a transaction from a passage"""
        tx_id = f"TX{random.randint(100000, 999999)}"
        
        # Base toll amount
        base_amount = self.toll_rates[passage.vehicle_class]
        
        # Apply random discounts (frequent user, off-peak, etc.)
        discount = 0
        if random.random() < 0.15:  # 15% chance of discount
            discount = random.uniform(0.05, 0.25)
        
        amount = base_amount * (1 - discount)
        
        # Transaction status (95% success rate)
        if random.random() < 0.95:
            status = TransactionStatus.SUCCESS
        elif random.random() < 0.03:
            status = TransactionStatus.FAILED
        elif random.random() < 0.015:
            status = TransactionStatus.DISPUTED
        else:
            status = TransactionStatus.PENDING
        
        payment_method = random.choice(self.payment_methods)
        toll_plaza_id = random.choice(self.toll_plazas)
        
        return Transaction(
            id=tx_id,
            passage_id=passage.id,
            timestamp=passage.timestamp,
            amount=amount,
            status=status,
            payment_method=payment_method,
            toll_plaza_id=toll_plaza_id,
            lane_id=passage.lane_id,
            vehicle_class=passage.vehicle_class,
            discount_applied=discount
        )
    
    def generate_ivdc_record(self, passage: Passage) -> IVDCRecord:
        """Generate IVDC validation record"""
        ivdc_id = f"IVDC{random.randint(100000, 999999)}"
        
        # Validation status (90% success rate)
        if random.random() < 0.90:
            validation_status = "VALID"
            error_code = None
            confidence_score = random.uniform(0.85, 0.99)
        else:
            validation_status = "INVALID"
            error_codes = ["OCR_FAILED", "CLASS_MISMATCH", "TAG_ERROR", "IMAGE_POOR"]
            error_code = random.choice(error_codes)
            confidence_score = random.uniform(0.3, 0.7)
        
        processing_time = random.uniform(0.1, 2.0)  # seconds
        
        return IVDCRecord(
            id=ivdc_id,
            timestamp=passage.timestamp + timedelta(milliseconds=random.randint(100, 500)),
            lane_id=passage.lane_id,
            validation_status=validation_status,
            error_code=error_code,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
    
    def generate_system_health(self) -> SystemHealth:
        """Generate system health data"""
        devices = [f"Device_{plaza}_{lane}" for plaza in self.toll_plazas for lane in range(1, 4)]
        device_id = random.choice(devices)
        
        # System status (95% online)
        if random.random() < 0.95:
            status = SystemStatus.ONLINE
            uptime_hours = random.uniform(100, 8760)  # Up to 1 year
            error_count = random.randint(0, 5)
        elif random.random() < 0.03:
            status = SystemStatus.MAINTENANCE
            uptime_hours = 0
            error_count = 0
        else:
            status = SystemStatus.OFFLINE
            uptime_hours = 0
            error_count = random.randint(5, 50)
        
        return SystemHealth(
            device_id=device_id,
            timestamp=datetime.now(),
            status=status,
            uptime_hours=uptime_hours,
            temperature=random.uniform(20, 60),
            memory_usage=random.uniform(0.3, 0.9),
            disk_usage=random.uniform(0.1, 0.8),
            network_latency=random.uniform(10, 200),
            error_count=error_count
        )
    
    def generate_pollution_data(self) -> Dict:
        """Generate air pollution monitoring data"""
        return {
            'timestamp': datetime.now(),
            'location': random.choice(self.toll_plazas),
            'pm25': random.uniform(10, 150),  # PM2.5 μg/m³
            'pm10': random.uniform(20, 200),   # PM10 μg/m³
            'no2': random.uniform(20, 80),     # NO2 μg/m³
            'co': random.uniform(0.5, 10),     # CO mg/m³
            'so2': random.uniform(5, 50),      # SO2 μg/m³
            'o3': random.uniform(50, 200),     # O3 μg/m³
            'noise_level': random.uniform(65, 85),  # dB
            'traffic_volume': random.randint(100, 1000)  # vehicles/hour
        }

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = TollingSystemSimulator()
    st.session_state.passages = []
    st.session_state.transactions = []
    st.session_state.ivdc_records = []
    st.session_state.system_health = []
    st.session_state.pollution_data = []

simulator = st.session_state.simulator

# Sidebar Navigation
st.sidebar.title("🛣️ Tolling System Playground")
st.sidebar.markdown("**Comprehensive Traffic & Revenue Analytics**")

# Page selection
page = st.sidebar.selectbox("Choose Analysis Module", [
    "🏠 Dashboard Overview",
    "🚗 Passage Explorer", 
    "💰 Transaction Analytics",
    "🔍 IVDC Validation",
    "📊 Traffic & Congestion",
    "🌍 Pollution Impact",
    "⚡ System Health",
    "🤖 ML Predictions",
    "🚀 Future Scope"
])

# Simulation Controls
st.sidebar.markdown("---")
st.sidebar.header("🎮 Simulation Controls")

# Generate data buttons
if st.sidebar.button("Generate Sample Data"):
    # Generate comprehensive sample data
    for _ in range(50):
        passage = simulator.generate_passage()
        transaction = simulator.generate_transaction(passage)
        ivdc = simulator.generate_ivdc_record(passage)
        health = simulator.generate_system_health()
        pollution = simulator.generate_pollution_data()
        
        st.session_state.passages.append(passage)
        st.session_state.transactions.append(transaction)
        st.session_state.ivdc_records.append(ivdc)
        st.session_state.system_health.append(health)
        st.session_state.pollution_data.append(pollution)
    st.sidebar.success("Generated 50 records for each data type!")
    st.rerun()

bulk_count = st.sidebar.slider("Bulk Generate Count", 10, 500, 100)
if st.sidebar.button("Generate Bulk Data"):
    for _ in range(bulk_count):
        passage = simulator.generate_passage()
        transaction = simulator.generate_transaction(passage)
        ivdc = simulator.generate_ivdc_record(passage)
        health = simulator.generate_system_health()
        pollution = simulator.generate_pollution_data()
        
        st.session_state.passages.append(passage)
        st.session_state.transactions.append(transaction)
        st.session_state.ivdc_records.append(ivdc)
        st.session_state.system_health.append(health)
        st.session_state.pollution_data.append(pollution)
    st.sidebar.success(f"Generated {bulk_count} records!")
    st.rerun()

if st.sidebar.button("Clear All Data"):
    st.session_state.passages = []
    st.session_state.transactions = []
    st.session_state.ivdc_records = []
    st.session_state.system_health = []
    st.session_state.pollution_data = []
    st.sidebar.success("All data cleared!")
    st.rerun()

# Data status
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Data Status:**")
st.sidebar.write(f"🚗 Passages: {len(st.session_state.passages)}")
st.sidebar.write(f"💰 Transactions: {len(st.session_state.transactions)}")
st.sidebar.write(f"🔍 IVDC Records: {len(st.session_state.ivdc_records)}")
st.sidebar.write(f"⚡ Health Records: {len(st.session_state.system_health)}")
st.sidebar.write(f"🌍 Pollution Records: {len(st.session_state.pollution_data)}")

# Main Content Area
if page == "🏠 Dashboard Overview":
    st.title("🛣️ Tolling System Dashboard Overview")
    
    if len(st.session_state.passages) > 0:
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_passages = len(st.session_state.passages)
        total_revenue = sum(tx.amount for tx in st.session_state.transactions if tx.status == TransactionStatus.SUCCESS)
        avg_speed = np.mean([p.speed for p in st.session_state.passages])
        success_rate = len([tx for tx in st.session_state.transactions if tx.status == TransactionStatus.SUCCESS]) / len(st.session_state.transactions) * 100
        systems_online = len([h for h in st.session_state.system_health if h.status == SystemStatus.ONLINE]) / len(st.session_state.system_health) * 100 if st.session_state.system_health else 0
        
        with col1:
            st.metric("Total Passages", f"{total_passages:,}")
        with col2:
            st.metric("Revenue Today", f"${total_revenue:,.2f}")
        with col3:
            st.metric("Avg Speed", f"{avg_speed:.1f} km/h")
        with col4:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col5:
            st.metric("Systems Online", f"{systems_online:.0f}%")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Traffic by Vehicle Class")
            class_counts = {}
            for passage in st.session_state.passages:
                class_counts[passage.vehicle_class.value] = class_counts.get(passage.vehicle_class.value, 0) + 1
            
            if class_counts:
                class_df = pd.DataFrame(list(class_counts.items()), columns=['Vehicle Class', 'Count'])
                st.bar_chart(class_df.set_index('Vehicle Class'))
        
        with col2:
            st.subheader("💰 Revenue by Payment Method")
            payment_revenue = {}
            for tx in st.session_state.transactions:
                if tx.status == TransactionStatus.SUCCESS:
                    payment_revenue[tx.payment_method] = payment_revenue.get(tx.payment_method, 0) + tx.amount
            
            if payment_revenue:
                payment_df = pd.DataFrame(list(payment_revenue.items()), columns=['Payment Method', 'Revenue'])
                st.bar_chart(payment_df.set_index('Payment Method'))
        
        # Recent Activity
        st.subheader("🕐 Recent Activity")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Latest Passages:**")
            recent_passages = sorted(st.session_state.passages, key=lambda x: x.timestamp, reverse=True)[:5]
            for passage in recent_passages:
                st.write(f"🚗 {passage.license_plate} | {passage.vehicle_class.value} | {passage.timestamp.strftime('%H:%M:%S')} | Lane {passage.lane_id}")
        
        with col2:
            st.write("**Recent Transactions:**")
            recent_transactions = sorted(st.session_state.transactions, key=lambda x: x.timestamp, reverse=True)[:5]
            for tx in recent_transactions:
                status_emoji = "✅" if tx.status == TransactionStatus.SUCCESS else "❌" if tx.status == TransactionStatus.FAILED else "⏳"
                st.write(f"{status_emoji} ${tx.amount:.2f} | {tx.payment_method} | {tx.timestamp.strftime('%H:%M:%S')}")
    
    else:
        st.info("👆 Generate sample data using the sidebar controls to see the dashboard in action!")
        
        st.markdown("""
        ## 🎯 System Capabilities
        
        This tolling system playground simulates:
        
        ### 📊 **Core Analytics**
        - **Passage Analysis**: Vehicle detection, classification, and flow monitoring
        - **Transaction Processing**: Revenue tracking, payment methods, success rates
        - **IVDC Validation**: Image/video data collection validation and error handling
        - **System Health**: Infrastructure monitoring and uptime tracking
        
        ### 🌟 **Advanced Features**
        - **Traffic Congestion Analysis**: Peak hours, lane utilization, bottleneck detection
        - **Environmental Impact**: Air pollution monitoring and emissions tracking
        - **Predictive Analytics**: ML models for traffic forecasting and revenue prediction
        - **Smart City Integration**: Future-ready features for connected infrastructure
        
        ### 🚀 **Getting Started**
        1. Use the sidebar to generate sample data
        2. Explore different analysis modules
        3. Experiment with traffic patterns and scenarios
        4. View real-time KPIs and insights
        """)

elif page == "🚗 Passage Explorer":
    st.title("🚗 Vehicle Passage Analysis")
    
    if len(st.session_state.passages) > 0:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            vehicle_filter = st.selectbox("Filter by Vehicle Class", 
                ["All"] + [v.value for v in VehicleClass])
        with col2:
            lane_filter = st.selectbox("Filter by Lane", ["All"] + simulator.lanes)
        with col3:
            min_speed = st.slider("Minimum Speed (km/h)", 0, 100, 0)
        
        # Filter passages
        filtered_passages = st.session_state.passages
        if vehicle_filter != "All":
            filtered_passages = [p for p in filtered_passages if p.vehicle_class.value == vehicle_filter]
        if lane_filter != "All":
            filtered_passages = [p for p in filtered_passages if p.lane_id == lane_filter]
        filtered_passages = [p for p in filtered_passages if p.speed >= min_speed]
        
        # Statistics
        st.subheader("📊 Passage Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Passages", len(filtered_passages))
        with col2:
            avg_speed = np.mean([p.speed for p in filtered_passages]) if filtered_passages else 0
            st.metric("Average Speed", f"{avg_speed:.1f} km/h")
        with col3:
            avg_length = np.mean([p.length for p in filtered_passages]) if filtered_passages else 0
            st.metric("Average Length", f"{avg_length:.1f} m")
        with col4:
            tag_rate = len([p for p in filtered_passages if p.tag_id]) / len(filtered_passages) * 100 if filtered_passages else 0
            st.metric("Tag Rate", f"{tag_rate:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Speed Distribution")
            if filtered_passages:
                speeds = [p.speed for p in filtered_passages]
                speed_hist, speed_bins = np.histogram(speeds, bins=20)
                speed_df = pd.DataFrame({
                    'Speed (km/h)': speed_bins[:-1],
                    'Count': speed_hist
                })
                st.bar_chart(speed_df.set_index('Speed (km/h)'))
        
        with col2:
            st.subheader("Lane Utilization")
            lane_counts = {}
            for passage in filtered_passages:
                lane_counts[passage.lane_id] = lane_counts.get(passage.lane_id, 0) + 1
            
            if lane_counts:
                lane_df = pd.DataFrame(list(lane_counts.items()), columns=['Lane', 'Passages'])
                st.bar_chart(lane_df.set_index('Lane'))
        
        # Detailed passage list
        st.subheader("📋 Passage Details")
        
        # Convert to DataFrame for better display
        passage_data = []
        for p in filtered_passages[:50]:  # Show latest 50
            passage_data.append({
                'ID': p.id,
                'Timestamp': p.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Lane': p.lane_id,
                'License Plate': p.license_plate,
                'Vehicle Class': p.vehicle_class.value,
                'Speed (km/h)': round(p.speed, 1),
                'Length (m)': round(p.length, 1),
                'Weight (kg)': round(p.weight, 0),
                'Tag ID': p.tag_id or "No Tag",
                'Image Quality': round(p.image_quality, 2)
            })
        
        if passage_data:
            passage_df = pd.DataFrame(passage_data)
            st.dataframe(passage_df, use_container_width=True)
    
    else:
        st.info("No passage data available. Generate data using the sidebar controls.")

elif page == "💰 Transaction Analytics":
    st.title("💰 Revenue & Transaction Analytics")
    
    if len(st.session_state.transactions) > 0:
        transactions = st.session_state.transactions
        
        # Revenue KPIs
        st.subheader("📈 Revenue Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue = sum(tx.amount for tx in transactions if tx.status == TransactionStatus.SUCCESS)
        total_transactions = len(transactions)
        success_rate = len([tx for tx in transactions if tx.status == TransactionStatus.SUCCESS]) / total_transactions * 100
        avg_transaction = total_revenue / len([tx for tx in transactions if tx.status == TransactionStatus.SUCCESS]) if any(tx.status == TransactionStatus.SUCCESS for tx in transactions) else 0
        
        with col1:
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col2:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col3:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col4:
            st.metric("Avg Transaction", f"${avg_transaction:.2f}")
        
        # Revenue analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue by Vehicle Class")
            class_revenue = {}
            for tx in transactions:
                if tx.status == TransactionStatus.SUCCESS:
                    class_revenue[tx.vehicle_class.value] = class_revenue.get(tx.vehicle_class.value, 0) + tx.amount
            
            if class_revenue:
                revenue_df = pd.DataFrame(list(class_revenue.items()), columns=['Vehicle Class', 'Revenue'])
                st.bar_chart(revenue_df.set_index('Vehicle Class'))
        
        with col2:
            st.subheader("Transaction Status Distribution")
            status_counts = {}
            for tx in transactions:
                status_counts[tx.status.value] = status_counts.get(tx.status.value, 0) + 1
            
            if status_counts:
                status_df = pd.DataFrame(list(status_counts.items()), columns=['Status', 'Count'])
                st.bar_chart(status_df.set_index('Status'))
        
        # Plaza performance
        st.subheader("🏢 Plaza Performance")
        plaza_stats = {}
        for tx in transactions:
            if tx.toll_plaza_id not in plaza_stats:
                plaza_stats[tx.toll_plaza_id] = {
                    'revenue': 0,
                    'transactions': 0,
                    'success': 0
                }
            plaza_stats[tx.toll_plaza_id]['transactions'] += 1
            if tx.status == TransactionStatus.SUCCESS:
                plaza_stats[tx.toll_plaza_id]['revenue'] += tx.amount
                plaza_stats[tx.toll_plaza_id]['success'] += 1
        
        plaza_data = []
        for plaza, stats in plaza_stats.items():
            success_rate = (stats['success'] / stats['transactions'] * 100) if stats['transactions'] > 0 else 0
            plaza_data.append({
                'Plaza': plaza,
                'Revenue': f"${stats['revenue']:.2f}",
                'Transactions': stats['transactions'],
                'Success Rate': f"{success_rate:.1f}%"
            })
        
        if plaza_data:
            plaza_df = pd.DataFrame(plaza_data)
            st.dataframe(plaza_df, use_container_width=True)
        
        # Hourly revenue trend
        st.subheader("⏰ Hourly Revenue Trend")
        hourly_revenue = {}
        for tx in transactions:
            if tx.status == TransactionStatus.SUCCESS:
                hour = tx.timestamp.strftime('%H:00')
                hourly_revenue[hour] = hourly_revenue.get(hour, 0) + tx.amount
        
        if hourly_revenue:
            # Sort by hour
            sorted_hours = sorted(hourly_revenue.items())
            hour_df = pd.DataFrame(sorted_hours, columns=['Hour', 'Revenue'])
            st.line_chart(hour_df.set_index('Hour'))
    
    else:
        st.info("No transaction data available. Generate data using the sidebar controls.")

elif page == "🔍 IVDC Validation":
    st.title("🔍 IVDC (Image/Video Data Collection) Analysis")
    
    if len(st.session_state.ivdc_records) > 0:
        ivdc_records = st.session_state.ivdc_records
        
        # IVDC KPIs
        st.subheader("📊 Validation Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        total_validations = len(ivdc_records)
        valid_count = len([r for r in ivdc_records if r.validation_status == "VALID"])
        #validation_rate = valid_count / total_validations *
