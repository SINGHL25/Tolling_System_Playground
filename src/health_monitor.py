"""
Health Monitor Module
====================

Monitors roadside system status, uptime, device health, and performance
for toll road infrastructure management.

Classes:
--------
- HealthMonitor: Main class for system health monitoring
- DeviceStatusTracker: Individual device status tracking
- AlertManager: Alert generation and management

Example Usage:
--------------
>>> monitor = HealthMonitor()
>>> health_status = monitor.check_system_health(device_data)
>>> alerts = monitor.generate_alerts(health_status)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

from .utils import DataProcessor, ValidationUtils, DateTimeUtils

# Configure logging
logger = logging.getLogger(__name__)

class DeviceStatus(Enum):
    """Enumeration for device status levels."""
    ONLINE = "Online"
    DEGRADED = "Degraded"
    OFFLINE = "Offline"
    MAINTENANCE = "Maintenance"
    ERROR = "Error"

class AlertSeverity(Enum):
    """Enumeration for alert severity levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class SystemHealthMetrics:
    """Data class for system health metrics."""
    total_devices: int
    online_devices: int
    degraded_devices: int
    offline_devices: int
    avg_uptime: float
    avg_response_time: float
    active_alerts: int
    system_availability: float

@dataclass
class DeviceHealthData:
    """Data class for individual device health data."""
    device_id: str
    status: DeviceStatus
    uptime_percent: float
    response_time_ms: float
    cpu_usage: float
    memory_usage: float
    temperature: float
    last_maintenance: datetime
    error_count: int

@dataclass
class Alert:
    """Data class for system alerts."""
    alert_id: str
    device_id: str
    severity: AlertSeverity
    alert_type: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

class HealthMonitor:
    """
    Main class for monitoring system health and device status.
    
    Attributes:
        config (dict): Configuration parameters
        processor (DataProcessor): Data processing utilities
        validator (ValidationUtils): Data validation utilities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize HealthMonitor.
        
        Args:
            config: Configuration dictionary with monitoring parameters
        """
        self.config = config or self._get_default_config()
        self.processor = DataProcessor()
        self.validator = ValidationUtils()
        self.datetime_utils = DateTimeUtils()
        
        # Initialize sub-components
        self.device_tracker = DeviceStatusTracker(self.config)
        self.alert_manager = AlertManager(self.config)
        
        logger.info("HealthMonitor initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'uptime_thresholds': {
                'excellent': 99.5,
                'good': 98.0,
                'acceptable': 95.0,
                'poor': 90.0
            },
            'performance_thresholds': {
                'response_time_ms': 1000,
                'cpu_usage_percent': 80,
                'memory_usage_percent': 85,
                'temperature_celsius': 70
            },
            'alert_thresholds': {
                'critical_uptime': 90.0,
                'critical_response_time': 5000,
                'high_error_count': 10
            },
            'maintenance_intervals': {
                'cameras': 30,      # days
                'sensors': 60,      # days
                'terminals': 45,    # days
                'servers': 90       # days
            },
            'data_retention_days': 90
        }
    
    def check_system_health(self, df: pd.DataFrame) -> SystemHealthMetrics:
        """
        Perform comprehensive system health check.
        
        Args:
            df: DataFrame with device health data
            
        Returns:
            SystemHealthMetrics object with health assessment
        """
        try:
            if df.empty:
                logger.warning("No device data available for health check")
                return self._empty_health_metrics()
            
            # Get latest status for each device
            latest_status = df.groupby('device_id').tail(1)
            
            # Count devices by status
            status_counts = latest_status['status'].value_counts()
            total_devices = len(latest_status)
            online_devices = status_counts.get('Online', 0)
            degraded_devices = status_counts.get('Degraded', 0)
            offline_devices = status_counts.get('Offline', 0)
            
            # Calculate average metrics
            avg_uptime = latest_status['uptime_percent'].mean()
            avg_response_time = latest_status[latest_status['status'] == 'Online']['response_time_ms'].mean()
            if pd.isna(avg_response_time):
                avg_response_time = 0
            
            # Count active alerts
            active_alerts = len(latest_status[
                (latest_status['uptime_percent'] < self.config['alert_thresholds']['critical_uptime']) |
                (latest_status['response_time_ms'] > self.config['alert_thresholds']['critical_response_time']) |
                (latest_status['error_count'] >= self.config['alert_thresholds']['high_error_count'])
            ])
            
            # System availability
            system_availability = (online_devices + degraded_devices) / total_devices * 100 if total_devices > 0 else 0
            
            metrics = SystemHealthMetrics(
                total_devices=total_devices,
                online_devices=online_devices,
                degraded_devices=degraded_devices,
                offline_devices=offline_devices,
                avg_uptime=avg_uptime,
                avg_response_time=avg_response_time,
                active_alerts=active_alerts,
                system_availability=system_availability
            )
            
            logger.info(f"System health check complete - {online_devices}/{total_devices} devices online")
            return metrics
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
            return self._empty_health_metrics()
    
    def analyze_device_performance(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze individual device performance metrics.
        
        Args:
            df: DataFrame with device performance data
            
        Returns:
            Dictionary with device performance analysis
        """
        try:
            if df.empty:
                return {}
            
            performance_analysis = {}
            
            # Performance by device category
            if 'category' in df.columns:
                category_performance = df.groupby('category').agg({
                    'uptime_percent': ['mean', 'std', 'min'],
                    'response_time_ms': ['mean', 'std', 'max'],
                    'cpu_usage_percent': ['mean', 'max'],
                    'temperature_celsius': ['mean', 'max']
                }).round(2)
                
                # Flatten column names
                category_performance.columns = ['_'.join(col).strip() for col in category_performance.columns]
                performance_analysis['category_performance'] = category_performance.to_dict('index')
            
            # Top performing devices
            latest_data = df.groupby('device_id').tail(1)
            top_performers = latest_data.nlargest(10, 'uptime_percent')[['device_id', 'uptime_percent', 'response_time_ms']]
            performance_analysis['top_performers'] = top_performers.to_dict('records')
            
            # Poor performing devices
            poor_performers = latest_data.nsmallest(10, 'uptime_percent')[['device_id', 'uptime_percent', 'status']]
            performance_analysis['poor_performers'] = poor_performers.to_dict('records')
            
            # Performance trends
            if 'timestamp' in df.columns:
                performance_trends = self._analyze_performance_trends(df)
                performance_analysis['trends'] = performance_trends
            
            logger.info("Device performance analysis completed")
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing device performance: {str(e)}")
            return {}
    
    def generate_alerts(self, df: pd.DataFrame) -> List[Alert]:
        """
        Generate alerts based on device health data.
        
        Args:
            df: DataFrame with current device health data
            
        Returns:
            List of Alert objects
        """
        try:
            if df.empty:
                return []
            
            alerts = []
            latest_data = df.groupby('device_id').tail(1)
            
            for _, device_data in latest_data.iterrows():
                device_id = device_data['device_id']
                
                # Uptime alerts
                uptime = device_data.get('uptime_percent', 100)
                if uptime < self.config['alert_thresholds']['critical_uptime']:
                    severity = AlertSeverity.CRITICAL if uptime < 50 else AlertSeverity.HIGH
                    alert = Alert(
                        alert_id=f"UPTIME_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        device_id=device_id,
                        severity=severity,
                        alert_type="Low Uptime",
                        message=f"Device {device_id} uptime is {uptime:.1f}% (threshold: {self.config['alert_thresholds']['critical_uptime']}%)",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # Response time alerts
                response_time = device_data.get('response_time_ms', 0)
                if response_time > self.config['alert_thresholds']['critical_response_time']:
                    alert = Alert(
                        alert_id=f"RESPONSE_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        device_id=device_id,
                        severity=AlertSeverity.HIGH,
                        alert_type="Slow Response",
                        message=f"Device {device_id} response time is {response_time:.0f}ms (threshold: {self.config['alert_thresholds']['critical_response_time']}ms)",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # Error count alerts
                error_count = device_data.get('error_count', 0)
                if error_count >= self.config['alert_thresholds']['high_error_count']:
                    alert = Alert(
                        alert_id=f"ERRORS_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        device_id=device_id,
                        severity=AlertSeverity.MEDIUM,
                        alert_type="High Error Count",
                        message=f"Device {device_id} has {error_count} errors (threshold: {self.config['alert_thresholds']['high_error_count']})",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # Temperature alerts
                temperature = device_data.get('temperature_celsius', 0)
                if temperature > self.config['performance_thresholds']['temperature_celsius']:
                    severity = AlertSeverity.CRITICAL if temperature > 80 else AlertSeverity.HIGH
                    alert = Alert(
                        alert_id=f"TEMP_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        device_id=device_id,
                        severity=severity,
                        alert_type="High Temperature",
                        message=f"Device {device_id} temperature is {temperature:.1f}°C (threshold: {self.config['performance_thresholds']['temperature_celsius']}°C)",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # CPU usage alerts
                cpu_usage = device_data.get('cpu_usage_percent', 0)
                if cpu_usage > self.config['performance_thresholds']['cpu_usage_percent']:
                    alert = Alert(
                        alert_id=f"CPU_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        device_id=device_id,
                        severity=AlertSeverity.MEDIUM,
                        alert_type="High CPU Usage",
                        message=f"Device {device_id} CPU usage is {cpu_usage:.1f}% (threshold: {self.config['performance_thresholds']['cpu_usage_percent']}%)",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                # Memory usage alerts
                memory_usage = device_data.get('memory_usage_percent', 0)
                if memory_usage > self.config['performance_thresholds']['memory_usage_percent']:
                    alert = Alert(
                        alert_id=f"MEMORY_{device_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        device_id=device_id,
                        severity=AlertSeverity.MEDIUM,
                        alert_type="High Memory Usage",
                        message=f"Device {device_id} memory usage is {memory_usage:.1f}% (threshold: {self.config['performance_thresholds']['memory_usage_percent']}%)",
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
            
            logger.info(f"Generated {len(alerts)} alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
            return []
    
    def check_maintenance_schedule(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check maintenance schedules and identify overdue devices.
        
        Args:
            df: DataFrame with device data including last maintenance dates
            
        Returns:
            Dictionary with maintenance schedule analysis
        """
        try:
            if df.empty or 'last_maintenance_date' not in df.columns:
                return {}
            
            maintenance_analysis = {}
            current_date = datetime.now().date()
            
            # Get latest data for each device
            latest_data = df.groupby('device_id').tail(1)
            
            overdue_devices = []
            due_soon_devices = []
            
            for _, device_data in latest_data.iterrows():
                device_id = device_data['device_id']
                category = device_data.get('category', 'unknown')
                last_maintenance = device_data['last_maintenance_date']
                
                if pd.isna(last_maintenance):
                    continue
                
                # Convert to date if datetime
                if isinstance(last_maintenance, datetime):
                    last_maintenance = last_maintenance.date()
                
                days_since_maintenance = (current_date - last_maintenance).days
                maintenance_interval = self.config['maintenance_intervals'].get(category.lower(), 60)
                
                if days_since_maintenance > maintenance_interval:
                    overdue_devices.append({
                        'device_id': device_id,
                        'category': category,
                        'days_overdue': days_since_maintenance - maintenance_interval,
                        'last_maintenance': last_maintenance.isoformat()
                    })
                elif days_since_maintenance > maintenance_interval * 0.8:  # 80% of interval
                    due_soon_devices.append({
                        'device_id': device_id,
                        'category': category,
                        'days_until_due': maintenance_interval - days_since_maintenance,
                        'last_maintenance': last_maintenance.isoformat()
                    })
            
            # Maintenance statistics by category
            maintenance_stats = latest_data.groupby('category').apply(
                lambda x: pd.Series({
                    'avg_days_since_maintenance': (current_date - pd.to_datetime(x['last_maintenance_date']).dt.date).dt.days.mean(),
                    'devices_overdue': sum(1 for _, row in x.iterrows() 
                                         if (current_date - pd.to_datetime(row['last_maintenance_date']).date()).days > 
                                         self.config['maintenance_intervals'].get(row['category'].lower(), 60)),
                    'total_devices': len(x)
                })
            ).round(1)
            
            maintenance_analysis = {
                'overdue_devices': overdue_devices,
                'due_soon_devices': due_soon_devices,
                'maintenance_stats': maintenance_stats.to_dict('index'),
                'total_overdue': len(overdue_devices),
                'total_due_soon': len(due_soon_devices)
            }
            
            logger.info(f"Maintenance check complete - {len(overdue_devices)} overdue, {len(due_soon_devices)} due soon")
            return maintenance_analysis
            
        except Exception as e:
            logger.error(f"Error checking maintenance schedule: {str(e)}")
            return {}
    
    def calculate_availability_metrics(self, df: pd.DataFrame,
                                     time_window_hours: int = 24) -> Dict[str, float]:
        """
        Calculate system availability metrics.
        
        Args:
            df: DataFrame with device status data
            time_window_hours: Time window for availability calculation
            
        Returns:
            Dictionary with availability metrics
        """
        try:
            if df.empty or 'timestamp' not in df.columns:
                return {}
            
            # Filter to specified time window
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            recent_data = df[df['timestamp'] >= cutoff_time]
            
            if recent_data.empty:
                return {}
            
            availability_metrics = {}
            
            # Overall system availability
            total_device_hours = len(recent_data)
            online_device_hours = len(recent_data[recent_data['status'] == 'Online'])
            degraded_device_hours = len(recent_data[recent_data['status'] == 'Degraded'])
            
            # Full availability (only online devices)
            full_availability = (online_device_hours / total_device_hours) * 100 if total_device_hours > 0 else 0
            
            # Partial availability (online + degraded devices)
            partial_availability = ((online_device_hours + degraded_device_hours) / total_device_hours) * 100 if total_device_hours > 0 else 0
            
            availability_metrics['full_availability_percent'] = full_availability
            availability_metrics['partial_availability_percent'] = partial_availability
            availability_metrics['time_window_hours'] = time_window_hours
            
            # Availability by device category
            if 'category' in recent_data.columns:
                category_availability = recent_data.groupby('category').apply(
                    lambda x: (len(x[x['status'] == 'Online']) / len(x)) * 100 if len(x) > 0 else 0
                ).round(2)
                availability_metrics['category_availability'] = category_availability.to_dict()
            
            # MTBF and MTTR calculations (simplified)
            mtbf_mttr = self._calculate_mtbf_mttr(recent_data)
            availability_metrics.update(mtbf_mttr)
            
            logger.info(f"Availability calculated: {full_availability:.2f}% full, {partial_availability:.2f}% partial")
            return availability_metrics
            
        except Exception as e:
            logger.error(f"Error calculating availability metrics: {str(e)}")
            return {}
    
    def generate_health_report(self, df: pd.DataFrame,
                             period_name: str = "System Health Report") -> Dict[str, any]:
        """
        Generate comprehensive system health report.
        
        Args:
            df: DataFrame with system health data
            period_name: Name of the reporting period
            
        Returns:
            Dictionary with complete health analysis
        """
        try:
            if df.empty:
                logger.warning("No data available for health report")
                return {}
            
            # Core analyses
            health_metrics = self.check_system_health(df)
            performance_analysis = self.analyze_device_performance(df)
            alerts = self.generate_alerts(df)
            maintenance_analysis = self.check_maintenance_schedule(df)
            availability_metrics = self.calculate_availability_metrics(df)
            
            # Summary statistics
            summary_stats = self._calculate_health_statistics(df)
            
            # Recommendations
            recommendations = self._generate_health_recommendations(health_metrics, alerts, maintenance_analysis)
            
            report = {
                'report_metadata': {
                    'period_name': period_name,
                    'generated_at': datetime.now().isoformat(),
                    'data_points': len(df),
                    'analysis_period': {
                        'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else 'N/A',
                        'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else 'N/A'
                    }
                },
                'health_metrics': health_metrics.__dict__,
                'performance_analysis': performance_analysis,
                'active_alerts': [alert.__dict__ for alert in alerts],
                'maintenance_analysis': maintenance_analysis,
                'availability_metrics': availability_metrics,
                'summary_statistics': summary_stats,
                'recommendations': recommendations
            }
            
            logger.info(f"Health report generated for {period_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating health report: {str(e)}")
            return {}
    
    # Helper methods
    
    def _empty_health_metrics(self) -> SystemHealthMetrics:
        """Return empty health metrics."""
        return SystemHealthMetrics(
            total_devices=0,
            online_devices=0,
            degraded_devices=0,
            offline_devices=0,
            avg_uptime=0,
            avg_response_time=0,
            active_alerts=0,
            system_availability=0
        )
    
    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze performance trends over time."""
        try:
            trends = {}
            
            # Uptime trends
            if 'uptime_percent' in df.columns:
                daily_uptime = df.groupby(df['timestamp'].dt.date)['uptime_percent'].mean()
                trends['uptime_trend'] = {
                    'slope': self._calculate_trend_slope(daily_uptime),
                    'current_avg': daily_uptime.tail(7).mean(),  # Last 7 days
                    'previous_avg': daily_uptime.head(-7).tail(7).mean() if len(daily_uptime) > 7 else daily_uptime.mean()
                }
            
            # Response time trends
            if 'response_time_ms' in df.columns:
                daily_response_time = df.groupby(df['timestamp'].dt.date)['response_time_ms'].mean()
                trends['response_time_trend'] = {
                    'slope': self._calculate_trend_slope(daily_response_time),
                    'current_avg': daily_response_time.tail(7).mean(),
                    'previous_avg': daily_response_time.head(-7).tail(7).mean() if len(daily_response_time) > 7 else daily_response_time.mean()
                }
            
            # Error trends
            if 'error_count' in df.columns:
                daily_errors = df.groupby(df['timestamp'].dt.date)['error_count'].sum()
                trends['error_trend'] = {
                    'slope': self._calculate_trend_slope(daily_errors),
                    'current_total': daily_errors.tail(7).sum(),
                    'previous_total': daily_errors.head(-7).tail(7).sum() if len(daily_errors) > 7 else daily_errors.sum()
                }
            
            return trends
            
        except Exception:
            return {}
    
    def _calculate_mtbf_mttr(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Mean Time Between Failures and Mean Time To Repair."""
        try:
            mtbf_mttr = {}
            
            # Simplified MTBF calculation
            if 'error_count' in df.columns:
                total_operating_time = len(df) * 15 / 60  # Assume 15-minute intervals, convert to hours
                total_failures = df['error_count'].sum()
                
                if total_failures > 0:
                    mtbf_hours = total_operating_time / total_failures
                    mtbf_mttr['mtbf_hours'] = mtbf_hours
                else:
                    mtbf_mttr['mtbf_hours'] = total_operating_time  # No failures
            
            # Simplified MTTR calculation
            offline_periods = df[df['status'] == 'Offline']
            if not offline_periods.empty:
                # Estimate repair time based on offline duration
                avg_offline_duration = 2  # Assume average 2 hours offline time
                mtbf_mttr['mttr_hours'] = avg_offline_duration
            else:
                mtbf_mttr['mttr_hours'] = 0
            
            return mtbf_mttr
            
        except Exception:
            return {}
    
    def _calculate_trend_slope(self, data: pd.Series) -> float:
        """Calculate trend slope using linear regression."""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            y = data.values
            
            # Remove NaN values
            valid_indices = ~np.isnan(y)
            if not valid_indices.any():
                return 0.0
            
            x_valid = x[valid_indices]
            y_valid = y[valid_indices]
            
            if len(x_valid) < 2:
                return 0.0
            
            # Simple linear regression
            n = len(x_valid)
            slope = (n * np.sum(x_valid * y_valid) - np.sum(x_valid) * np.sum(y_valid)) / (n * np.sum(x_valid**2) - (np.sum(x_valid))**2)
            
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_health_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate summary health statistics."""
        try:
            stats = {}
            
            # Device status distribution
            if 'status' in df.columns:
                latest_status = df.groupby('device_id').tail(1)
                status_dist = latest_status['status'].value_counts(normalize=True) * 100
                stats['status_distribution'] = status_dist.to_dict()
            
            # Performance statistics
            if 'uptime_percent' in df.columns:
                stats['uptime_stats'] = {
                    'mean': df['uptime_percent'].mean(),
                    'median': df['uptime_percent'].median(),
                    'min': df['uptime_percent'].min(),
                    'max': df['uptime_percent'].max(),
                    'std': df['uptime_percent'].std()
                }
            
            if 'response_time_ms' in df.columns:
                online_devices = df[df['status'] == 'Online']
                if not online_devices.empty:
                    stats['response_time_stats'] = {
                        'mean': online_devices['response_time_ms'].mean(),
                        'median': online_devices['response_time_ms'].median(),
                        'p95': online_devices['response_time_ms'].quantile(0.95),
                        'max': online_devices['response_time_ms'].max()
                    }
            
            # Error statistics
            if 'error_count' in df.columns:
                stats['error_stats'] = {
                    'total_errors': df['error_count'].sum(),
                    'avg_errors_per_device': df.groupby('device_id')['error_count'].sum().mean(),
                    'devices_with_errors': (df.groupby('device_id')['error_count'].sum() > 0).sum()
                }
            
            return stats
            
        except Exception:
            return {}
    
    def _generate_health_recommendations(self, health_metrics: SystemHealthMetrics,
                                       alerts: List[Alert],
                                       maintenance_analysis: Dict[str, any]) -> List[str]:
        """Generate health management recommendations."""
        recommendations = []
        
        try:
            # System availability recommendations
            if health_metrics.system_availability < 95:
                recommendations.append(f"System availability is {health_metrics.system_availability:.1f}% - investigate offline devices and implement redundancy")
            
            # Device performance recommendations
            if health_metrics.avg_uptime < 98:
                recommendations.append(f"Average uptime is {health_metrics.avg_uptime:.1f}% - focus on improving device reliability")
            
            if health_metrics.avg_response_time > 1000:
                recommendations.append(f"Average response time is {health_metrics.avg_response_time:.0f}ms - optimize system performance")
            
            # Alert-based recommendations
            critical_alerts = [alert for alert in alerts if alert.severity == AlertSeverity.CRITICAL]
            if critical_alerts:
                recommendations.append(f"Address {len(critical_alerts)} critical alerts immediately to prevent system failures")
            
            high_alerts = [alert for alert in alerts if alert.severity == AlertSeverity.HIGH]
            if len(high_alerts) > 5:
                recommendations.append(f"High number of alerts ({len(high_alerts)}) indicates systemic issues - conduct comprehensive review")
            
            # Maintenance recommendations
            overdue_count = maintenance_analysis.get('total_overdue', 0)
            if overdue_count > 0:
                recommendations.append(f"{overdue_count} devices are overdue for maintenance - schedule immediate maintenance to prevent failures")
            
            due_soon_count = maintenance_analysis.get('total_due_soon', 0)
            if due_soon_count > 5:
                recommendations.append(f"{due_soon_count} devices need maintenance soon - plan maintenance schedule to avoid service disruption")
            
            # Offline device recommendations
            if health_metrics.offline_devices > 0:
                recommendations.append(f"{health_metrics.offline_devices} devices are offline - investigate and restore service immediately")
            
            # Degraded device recommendations
            if health_metrics.degraded_devices > health_metrics.total_devices * 0.1:  # More than 10% degraded
                recommendations.append("High number of degraded devices - investigate common issues and implement preventive measures")
            
            # Positive recommendations
            if health_metrics.system_availability >= 99 and health_metrics.avg_uptime >= 99:
                recommendations.append("Excellent system health - maintain current monitoring and maintenance practices")
            
            # Default recommendation if none generated
            if not recommendations:
                recommendations.append("System health is within acceptable parameters - continue regular monitoring")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
            recommendations.append("Manual health assessment recommended - automated analysis unavailable")
        
        return recommendations


class DeviceStatusTracker:
    """
    Specialized tracker for individual device status monitoring.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.datetime_utils = DateTimeUtils()
    
    def track_device_status(self, device_id: str, df: pd.DataFrame) -> DeviceHealthData:
        """
        Track status for a specific device.
        
        Args:
            device_id: Unique device identifier
            df: DataFrame with device health data
            
        Returns:
            DeviceHealthData object
        """
        try:
            device_data = df[df['device_id'] == device_id]
            
            if device_data.empty:
                logger.warning(f"No data found for device {device_id}")
                return self._empty_device_health_data(device_id)
            
            # Get latest status
            latest = device_data.tail(1).iloc[0]
            
            # Determine device status
            status = self._determine_device_status(latest)
            
            # Calculate metrics
            uptime_percent = latest.get('uptime_percent', 0)
            response_time_ms = latest.get('response_time_ms', 0)
            cpu_usage = latest.get('cpu_usage_percent', 0)
            memory_usage = latest.get('memory_usage_percent', 0)
            temperature = latest.get('temperature_celsius', 0)
            error_count = latest.get('error_count', 0)
            
            # Last maintenance
            last_maintenance = latest.get('last_maintenance_date')
            if pd.isna(last_maintenance):
                last_maintenance = datetime.now() - timedelta(days=365)  # Default to 1 year ago
            elif isinstance(last_maintenance, str):
                last_maintenance = datetime.fromisoformat(last_maintenance)
            
            health_data = DeviceHealthData(
                device_id=device_id,
                status=status,
                uptime_percent=uptime_percent,
                response_time_ms=response_time_ms,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                temperature=temperature,
                last_maintenance=last_maintenance,
                error_count=error_count
            )
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error tracking device {device_id}: {str(e)}")
            return self._empty_device_health_data(device_id)
    
    def _determine_device_status(self, device_data: pd.Series) -> DeviceStatus:
        """Determine device status based on metrics."""
        try:
            uptime = device_data.get('uptime_percent', 0)
            response_time = device_data.get('response_time_ms', 0)
            error_count = device_data.get('error_count', 0)
            status_str = device_data.get('status', '').upper()
            
            # Check for explicit maintenance status
            if status_str == 'MAINTENANCE':
                return DeviceStatus.MAINTENANCE
            
            # Check for offline status
            if uptime < 50 or status_str == 'OFFLINE':
                return DeviceStatus.OFFLINE
            
            # Check for error conditions
            if error_count >= self.config['alert_thresholds']['high_error_count']:
                return DeviceStatus.ERROR
            
            # Check for degraded performance
            if (uptime < self.config['uptime_thresholds']['acceptable'] or
                response_time > self.config['performance_thresholds']['response_time_ms']):
                return DeviceStatus.DEGRADED
            
            # Default to online
            return DeviceStatus.ONLINE
            
        except Exception:
            return DeviceStatus.ERROR
    
    def _empty_device_health_data(self, device_id: str) -> DeviceHealthData:
        """Return empty device health data."""
        return DeviceHealthData(
            device_id=device_id,
            status=DeviceStatus.ERROR,
            uptime_percent=0,
            response_time_ms=0,
            cpu_usage=0,
            memory_usage=0,
            temperature=0,
            last_maintenance=datetime.now() - timedelta(days=365),
            error_count=999
        )


class AlertManager:
    """
    Specialized manager for system alerts and notifications.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.active_alerts = []
    
    def process_alerts(self, alerts: List[Alert]) -> Dict[str, any]:
        """
        Process and categorize alerts.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            Dictionary with alert analysis
        """
        try:
            if not alerts:
                return {'total_alerts': 0}
            
            # Categorize alerts by severity
            severity_counts = {}
            for severity in AlertSeverity:
                severity_counts[severity.value] = len([a for a in alerts if a.severity == severity])
            
            # Categorize alerts by type
            type_counts = {}
            for alert in alerts:
                alert_type = alert.alert_type
                type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            
            # Device alert counts
            device_counts = {}
            for alert in alerts:
                device_id = alert.device_id
                device_counts[device_id] = device_counts.get(device_id, 0) + 1
            
            # Top problematic devices
            top_devices = sorted(device_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            alert_analysis = {
                'total_alerts': len(alerts),
                'severity_distribution': severity_counts,
                'alert_type_distribution': type_counts,
                'device_alert_counts': device_counts,
                'top_problematic_devices': dict(top_devices),
                'critical_alerts': [a.__dict__ for a in alerts if a.severity == AlertSeverity.CRITICAL],
                'latest_alerts': [a.__dict__ for a in sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:20]]
            }
            
            return alert_analysis
            
        except Exception as e:
            logger.error(f"Error processing alerts: {str(e)}")
            return {'total_alerts': 0}
    
    def generate_alert_summary(self, alerts: List[Alert]) -> str:
        """
        Generate human-readable alert summary.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            String with alert summary
        """
        try:
            if not alerts:
                return "No active alerts"
            
            critical_count = len([a for a in alerts if a.severity == AlertSeverity.CRITICAL])
            high_count = len([a for a in alerts if a.severity == AlertSeverity.HIGH])
            medium_count = len([a for a in alerts if a.severity == AlertSeverity.MEDIUM])
            low_count = len([a for a in alerts if a.severity == AlertSeverity.LOW])
            
            summary_parts = []
            
            if critical_count > 0:
                summary_parts.append(f"{critical_count} critical")
            if high_count > 0:
                summary_parts.append(f"{high_count} high")
            if medium_count > 0:
                summary_parts.append(f"{medium_count} medium")
            if low_count > 0:
                summary_parts.append(f"{low_count} low")
            
            severity_summary = ", ".join(summary_parts)
            
            # Most common alert type
            type_counts = {}
            for alert in alerts:
                alert_type = alert.alert_type
                type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "Unknown"
            
            summary = f"Total: {len(alerts)} alerts ({severity_summary}). Most common: {most_common_type}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating alert summary: {str(e)}")
            return f"Error generating summary for {len(alerts)} alerts"
    
    def prioritize_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """
        Prioritize alerts based on severity and impact.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            List of alerts sorted by priority
        """
        try:
            # Define severity priority
            severity_priority = {
                AlertSeverity.CRITICAL: 4,
                AlertSeverity.HIGH: 3,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.LOW: 1
            }
            
            # Sort by severity (highest first) and then by timestamp (newest first)
            prioritized_alerts = sorted(
                alerts,
                key=lambda x: (severity_priority.get(x.severity, 0), x.timestamp),
                reverse=True
            )
            
            return prioritized_alerts
            
        except Exception as e:
            logger.error(f"Error prioritizing alerts: {str(e)}")
            return alerts
