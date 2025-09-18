"""
Congestion Estimator Module
===========================

Calculates traffic congestion KPIs, wait times, and flow metrics
for toll road traffic management and optimization.

Classes:
--------
- CongestionEstimator: Main class for congestion analysis
- TrafficFlowAnalyzer: Traffic flow metrics and patterns
- WaitTimeCalculator: Queue and wait time calculations

Example Usage:
--------------
>>> estimator = CongestionEstimator()
>>> congestion_metrics = estimator.calculate_congestion_metrics(traffic_data)
>>> wait_times = estimator.estimate_wait_times(queue_data)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import warnings

from .utils import DataProcessor, ValidationUtils, DateTimeUtils

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CongestionMetrics:
    """Data class for congestion analysis metrics."""
    congestion_level: str
    avg_speed: float
    occupancy_percent: float
    delay_minutes: float
    throughput_vph: float
    travel_time_index: float
    queue_length: int
    density_vpm: float

@dataclass
class WaitTimeMetrics:
    """Data class for wait time analysis."""
    avg_wait_time: float
    max_wait_time: float
    wait_time_95th_percentile: float
    service_rate: float
    queue_efficiency: float

class CongestionEstimator:
    """
    Main class for estimating traffic congestion and calculating related KPIs.
    
    Attributes:
        config (dict): Configuration parameters
        processor (DataProcessor): Data processing utilities
        validator (ValidationUtils): Data validation utilities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CongestionEstimator.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._get_default_config()
        self.processor = DataProcessor()
        self.validator = ValidationUtils()
        self.datetime_utils = DateTimeUtils()
        
        logger.info("CongestionEstimator initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'lane_capacity_vph': 1800,  # vehicles per hour
            'free_flow_speed': 65,      # mph
            'congestion_thresholds': {
                'light': 0.3,           # 30% occupancy
                'moderate': 0.6,        # 60% occupancy
                'heavy': 0.8            # 80% occupancy
            },
            'speed_thresholds': {
                'free_flow': 55,        # mph
                'congested': 35,        # mph
                'heavy_congestion': 20  # mph
            },
            'analysis_interval_minutes': 15,
            'smoothing_window': 3
        }
    
    def calculate_congestion_metrics(self, df: pd.DataFrame,
                                   lane_id: Optional[str] = None) -> CongestionMetrics:
        """
        Calculate comprehensive congestion metrics.
        
        Args:
            df: DataFrame with traffic data
            lane_id: Specific lane to analyze (optional)
            
        Returns:
            CongestionMetrics object with analysis results
        """
        try:
            # Filter by lane if specified
            if lane_id and 'lane_id' in df.columns:
                df = df[df['lane_id'] == lane_id]
            
            if df.empty:
                logger.warning("No data available for congestion analysis")
                return self._empty_congestion_metrics()
            
            # Calculate basic metrics
            avg_speed = df['avg_speed_mph'].mean() if 'avg_speed_mph' in df.columns else 0
            occupancy_percent = df['occupancy_percent'].mean() if 'occupancy_percent' in df.columns else 0
            
            # Determine congestion level
            congestion_level = self._classify_congestion_level(occupancy_percent, avg_speed)
            
            # Calculate delay
            free_flow_speed = self.config['free_flow_speed']
            delay_minutes = self._calculate_delay(avg_speed, free_flow_speed)
            
            # Calculate throughput
            throughput_vph = df['throughput_vph'].mean() if 'throughput_vph' in df.columns else self._estimate_throughput(df)
            
            # Travel time index (1.0 = free flow conditions)
            travel_time_index = free_flow_speed / max(avg_speed, 1) if avg_speed > 0 else 1.0
            
            # Queue metrics
            queue_length = int(df['queue_length_vehicles'].mean()) if 'queue_length_vehicles' in df.columns else 0
            
            # Traffic density (vehicles per mile)
            density_vpm = throughput_vph / max(avg_speed, 1) if avg_speed > 0 else 0
            
            metrics = CongestionMetrics(
                congestion_level=congestion_level,
                avg_speed=avg_speed,
                occupancy_percent=occupancy_percent,
                delay_minutes=delay_minutes,
                throughput_vph=throughput_vph,
                travel_time_index=travel_time_index,
                queue_length=queue_length,
                density_vpm=density_vpm
            )
            
            logger.info(f"Congestion analysis complete - Level: {congestion_level}, Speed: {avg_speed:.1f} mph")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating congestion metrics: {str(e)}")
            return self._empty_congestion_metrics()
    
    def analyze_congestion_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze congestion patterns over time and by location.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            Dictionary with congestion pattern analysis
        """
        try:
            if df.empty:
                return {}
            
            patterns = {}
            
            # Hourly congestion patterns
            if 'timestamp' in df.columns:
                hourly_congestion = self._analyze_hourly_patterns(df)
                patterns['hourly_patterns'] = hourly_congestion
            
            # Daily patterns (weekday vs weekend)
            if 'timestamp' in df.columns:
                daily_patterns = self._analyze_daily_patterns(df)
                patterns['daily_patterns'] = daily_patterns
            
            # Lane-specific patterns
            if 'lane_id' in df.columns:
                lane_patterns = self._analyze_lane_patterns(df)
                patterns['lane_patterns'] = lane_patterns
            
            # Seasonal patterns (if data spans multiple months)
            if 'timestamp' in df.columns:
                seasonal_patterns = self._analyze_seasonal_patterns(df)
                patterns['seasonal_patterns'] = seasonal_patterns
            
            # Peak period identification
            peak_periods = self._identify_peak_periods(df)
            patterns['peak_periods'] = peak_periods
            
            logger.info("Congestion pattern analysis completed")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing congestion patterns: {str(e)}")
            return {}
    
    def estimate_wait_times(self, df: pd.DataFrame) -> WaitTimeMetrics:
        """
        Estimate wait times and queue dynamics.
        
        Args:
            df: DataFrame with queue and service data
            
        Returns:
            WaitTimeMetrics object with wait time analysis
        """
        try:
            if df.empty or 'avg_wait_time_sec' not in df.columns:
                logger.warning("No wait time data available")
                return WaitTimeMetrics(0, 0, 0, 0, 0)
            
            wait_times = df['avg_wait_time_sec']
            
            # Basic wait time statistics
            avg_wait_time = wait_times.mean()
            max_wait_time = wait_times.max()
            wait_time_95th_percentile = wait_times.quantile(0.95)
            
            # Service rate calculation
            service_rate = self._calculate_service_rate(df)
            
            # Queue efficiency
            queue_efficiency = self._calculate_queue_efficiency(df)
            
            metrics = WaitTimeMetrics(
                avg_wait_time=avg_wait_time,
                max_wait_time=max_wait_time,
                wait_time_95th_percentile=wait_time_95th_percentile,
                service_rate=service_rate,
                queue_efficiency=queue_efficiency
            )
            
            logger.info(f"Wait time analysis complete - Average: {avg_wait_time:.1f} seconds")
            return metrics
            
        except Exception as e:
            logger.error(f"Error estimating wait times: {str(e)}")
            return WaitTimeMetrics(0, 0, 0, 0, 0)
    
    def calculate_level_of_service(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate Level of Service (LOS) metrics based on HCM standards.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            Dictionary with LOS analysis
        """
        try:
            los_analysis = {}
            
            # Speed-based LOS
            if 'avg_speed_mph' in df.columns:
                speed_los = self._calculate_speed_based_los(df['avg_speed_mph'])
                los_analysis['speed_based_los'] = speed_los
            
            # Density-based LOS
            if 'density_vpm' in df.columns:
                density_los = self._calculate_density_based_los(df['density_vpm'])
                los_analysis['density_based_los'] = density_los
            
            # Volume-to-capacity ratio
            if 'throughput_vph' in df.columns:
                vc_ratio = df['throughput_vph'].mean() / self.config['lane_capacity_vph']
                vc_los = self._calculate_vc_based_los(vc_ratio)
                los_analysis['volume_capacity_los'] = vc_los
                los_analysis['vc_ratio'] = vc_ratio
            
            # Overall LOS (worst case)
            los_grades = [los_analysis.get(key, {}).get('grade', 'F') for key in ['speed_based_los', 'density_based_los', 'volume_capacity_los']]
            los_grades = [grade for grade in los_grades if grade != 'F']
            
            if los_grades:
                # Convert grades to numbers, find worst, convert back
                grade_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
                num_to_grade = {v: k for k, v in grade_to_num.items()}
                worst_grade_num = max([grade_to_num.get(grade, 6) for grade in los_grades])
                overall_los = num_to_grade[worst_grade_num]
            else:
                overall_los = 'F'
            
            los_analysis['overall_los'] = {
                'grade': overall_los,
                'description': self._get_los_description(overall_los)
            }
            
            logger.info(f"LOS analysis complete - Overall grade: {overall_los}")
            return los_analysis
            
        except Exception as e:
            logger.error(f"Error calculating LOS: {str(e)}")
            return {}
    
    def detect_incidents(self, df: pd.DataFrame, 
                        sensitivity: float = 2.0) -> List[Dict[str, any]]:
        """
        Detect potential traffic incidents based on anomalous patterns.
        
        Args:
            df: DataFrame with traffic data
            sensitivity: Sensitivity threshold for incident detection
            
        Returns:
            List of detected incidents
        """
        try:
            incidents = []
            
            if df.empty or 'timestamp' not in df.columns:
                return incidents
            
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            
            # Speed-based incident detection
            if 'avg_speed_mph' in df_sorted.columns:
                speed_incidents = self._detect_speed_incidents(df_sorted, sensitivity)
                incidents.extend(speed_incidents)
            
            # Volume-based incident detection
            if 'vehicle_count' in df_sorted.columns:
                volume_incidents = self._detect_volume_incidents(df_sorted, sensitivity)
                incidents.extend(volume_incidents)
            
            # Queue length incidents
            if 'queue_length_vehicles' in df_sorted.columns:
                queue_incidents = self._detect_queue_incidents(df_sorted, sensitivity)
                incidents.extend(queue_incidents)
            
            # Remove duplicates and sort by timestamp
            incidents = self._deduplicate_incidents(incidents)
            incidents.sort(key=lambda x: x['timestamp'])
            
            logger.info(f"Detected {len(incidents)} potential incidents")
            return incidents
            
        except Exception as e:
            logger.error(f"Error detecting incidents: {str(e)}")
            return []
    
    def calculate_economic_impact(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate the economic impact of congestion.
        
        Args:
            df: DataFrame with traffic and congestion data
            
        Returns:
            Dictionary with economic impact metrics
        """
        try:
            if df.empty:
                return {}
            
            # Constants for economic calculations (USD)
            VALUE_OF_TIME_PER_HOUR = 25.0  # Average value of time
            FUEL_COST_PER_GALLON = 3.50
            VEHICLE_OPERATING_COST_PER_MILE = 0.56
            
            economic_impact = {}
            
            # Time delay costs
            if 'delay_minutes' in df.columns and 'vehicle_count' in df.columns:
                total_delay_hours = (df['delay_minutes'] * df['vehicle_count']).sum() / 60
                time_cost = total_delay_hours * VALUE_OF_TIME_PER_HOUR
                economic_impact['time_delay_cost'] = time_cost
            
            # Fuel consumption costs due to congestion
            if 'avg_speed_mph' in df.columns and 'vehicle_count' in df.columns:
                fuel_waste_cost = self._calculate_fuel_waste_cost(df, FUEL_COST_PER_GALLON)
                economic_impact['fuel_waste_cost'] = fuel_waste_cost
            
            # Vehicle operating costs
            if 'vehicle_count' in df.columns:
                # Assume average trip length and additional wear due to congestion
                avg_trip_length = 10  # miles
                congestion_factor = 1.2  # 20% additional wear
                operating_cost = df['vehicle_count'].sum() * avg_trip_length * VEHICLE_OPERATING_COST_PER_MILE * (congestion_factor - 1)
                economic_impact['additional_operating_cost'] = operating_cost
            
            # Environmental costs (simplified)
            if 'co2_emissions_kg' in df.columns:
                co2_cost_per_kg = 0.05  # Social cost of carbon
                environmental_cost = df['co2_emissions_kg'].sum() * co2_cost_per_kg
                economic_impact['environmental_cost'] = environmental_cost
            
            # Total economic impact
            total_impact = sum(economic_impact.values())
            economic_impact['total_impact'] = total_impact
            
            # Calculate per-vehicle impact
            if 'vehicle_count' in df.columns and df['vehicle_count'].sum() > 0:
                economic_impact['cost_per_vehicle'] = total_impact / df['vehicle_count'].sum()
            
            logger.info(f"Economic impact calculated: ${total_impact:,.2f} total cost")
            return economic_impact
            
        except Exception as e:
            logger.error(f"Error calculating economic impact: {str(e)}")
            return {}
    
    def generate_congestion_report(self, df: pd.DataFrame,
                                 period_name: str = "Analysis Period") -> Dict[str, any]:
        """
        Generate comprehensive congestion analysis report.
        
        Args:
            df: DataFrame with traffic data
            period_name: Name of the analysis period
            
        Returns:
            Dictionary with complete congestion analysis
        """
        try:
            if df.empty:
                logger.warning("No data available for congestion report")
                return {}
            
            # Core analyses
            congestion_metrics = self.calculate_congestion_metrics(df)
            wait_time_metrics = self.estimate_wait_times(df)
            congestion_patterns = self.analyze_congestion_patterns(df)
            los_analysis = self.calculate_level_of_service(df)
            incidents = self.detect_incidents(df)
            economic_impact = self.calculate_economic_impact(df)
            
            # Summary statistics
            summary_stats = self._calculate_summary_statistics(df)
            
            # Recommendations
            recommendations = self._generate_congestion_recommendations(df, congestion_metrics, los_analysis)
            
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
                'congestion_metrics': congestion_metrics.__dict__,
                'wait_time_metrics': wait_time_metrics.__dict__,
                'congestion_patterns': congestion_patterns,
                'level_of_service': los_analysis,
                'incidents_detected': len(incidents),
                'incident_details': incidents,
                'economic_impact': economic_impact,
                'summary_statistics': summary_stats,
                'recommendations': recommendations
            }
            
            logger.info(f"Congestion report generated for {period_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating congestion report: {str(e)}")
            return {}
    
    # Helper methods
    
    def _empty_congestion_metrics(self) -> CongestionMetrics:
        """Return empty congestion metrics."""
        return CongestionMetrics(
            congestion_level="Unknown",
            avg_speed=0,
            occupancy_percent=0,
            delay_minutes=0,
            throughput_vph=0,
            travel_time_index=1.0,
            queue_length=0,
            density_vpm=0
        )
    
    def _classify_congestion_level(self, occupancy: float, speed: float) -> str:
        """Classify congestion level based on occupancy and speed."""
        thresholds = self.config['congestion_thresholds']
        speed_thresholds = self.config['speed_thresholds']
        
        # Primary classification by occupancy
        if occupancy < thresholds['light']:
            congestion_level = "Free Flow"
        elif occupancy < thresholds['moderate']:
            congestion_level = "Light"
        elif occupancy < thresholds['heavy']:
            congestion_level = "Moderate"
        else:
            congestion_level = "Heavy"
        
        # Adjust based on speed if available
        if speed > 0:
            if speed < speed_thresholds['heavy_congestion']:
                congestion_level = "Heavy"
            elif speed < speed_thresholds['congested'] and congestion_level in ["Free Flow", "Light"]:
                congestion_level = "Moderate"
        
        return congestion_level
    
    def _calculate_delay(self, current_speed: float, free_flow_speed: float) -> float:
        """Calculate delay in minutes based on speed reduction."""
        if current_speed <= 0 or free_flow_speed <= 0:
            return 0
        
        # Assume a standard trip length for delay calculation
        trip_length_miles = 10
        
        free_flow_time = (trip_length_miles / free_flow_speed) * 60  # minutes
        current_time = (trip_length_miles / current_speed) * 60      # minutes
        
        delay = max(0, current_time - free_flow_time)
        return delay
    
    def _estimate_throughput(self, df: pd.DataFrame) -> float:
        """Estimate throughput when not directly available."""
        if 'vehicle_count' in df.columns and 'timestamp' in df.columns:
            # Calculate vehicles per hour based on data frequency
            time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            if time_span_hours > 0:
                return df['vehicle_count'].sum() / time_span_hours
        
        return 0.0
    
    def _analyze_hourly_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze congestion patterns by hour of day."""
        hourly_data = df.groupby(df['timestamp'].dt.hour).agg({
            'occupancy_percent': 'mean',
            'avg_speed_mph': 'mean',
            'vehicle_count': 'sum' if 'vehicle_count' in df.columns else 'count'
        }).round(2)
        
        # Identify peak hours
        peak_occupancy_hour = hourly_data['occupancy_percent'].idxmax()
        peak_volume_hour = hourly_data.iloc[:, 2].idxmax()  # vehicle_count column
        
        return {
            'hourly_data': hourly_data.to_dict('index'),
            'peak_occupancy_hour': peak_occupancy_hour,
            'peak_volume_hour': peak_volume_hour
        }
    
    def _analyze_daily_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze congestion patterns by day of week."""
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.weekday >= 5
        
        daily_data = df.groupby('day_of_week').agg({
            'occupancy_percent': 'mean',
            'avg_speed_mph': 'mean',
            'vehicle_count': 'sum' if 'vehicle_count' in df.columns else 'count'
        }).round(2)
        
        # Weekend vs weekday comparison
        weekend_data = df[df['is_weekend']].agg({
            'occupancy_percent': 'mean',
            'avg_speed_mph': 'mean'
        })
        
        weekday_data = df[~df['is_weekend']].agg({
            'occupancy_percent': 'mean',
            'avg_speed_mph': 'mean'
        })
        
        return {
            'daily_data': daily_data.to_dict('index'),
            'weekend_vs_weekday': {
                'weekend': weekend_data.to_dict(),
                'weekday': weekday_data.to_dict()
            }
        }
    
    def _analyze_lane_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze congestion patterns by lane."""
        lane_data = df.groupby('lane_id').agg({
            'occupancy_percent': ['mean', 'std'],
            'avg_speed_mph': ['mean', 'std'],
            'vehicle_count': 'sum' if 'vehicle_count' in df.columns else 'count'
        }).round(2)
        
        # Flatten column names
        lane_data.columns = ['_'.join(col).strip() for col in lane_data.columns]
        
        # Identify problematic lanes
        high_congestion_lanes = lane_data[lane_data['occupancy_percent_mean'] > 70].index.tolist()
        low_speed_lanes = lane_data[lane_data['avg_speed_mph_mean'] < 35].index.tolist()
        
        return {
            'lane_data': lane_data.to_dict('index'),
            'high_congestion_lanes': high_congestion_lanes,
            'low_speed_lanes': low_speed_lanes
        }
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze seasonal congestion patterns."""
        df['month'] = df['timestamp'].dt.month
        
        monthly_data = df.groupby('month').agg({
            'occupancy_percent': 'mean',
            'avg_speed_mph': 'mean',
            'vehicle_count': 'sum' if 'vehicle_count' in df.columns else 'count'
        }).round(2)
        
        # Season mapping
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        
        df['season'] = df['month'].map(season_map)
        seasonal_data = df.groupby('season').agg({
            'occupancy_percent': 'mean',
            'avg_speed_mph': 'mean',
            'vehicle_count': 'sum' if 'vehicle_count' in df.columns else 'count'
        }).round(2)
        
        return {
            'monthly_data': monthly_data.to_dict('index'),
            'seasonal_data': seasonal_data.to_dict('index')
        }
    
    def _identify_peak_periods(self, df: pd.DataFrame) -> Dict[str, any]:
        """Identify recurring peak congestion periods."""
        if 'timestamp' not in df.columns:
            return {}
        
        # Calculate hourly occupancy averages
        hourly_occupancy = df.groupby(df['timestamp'].dt.hour)['occupancy_percent'].mean()
        
        # Define peak threshold (75th percentile)
        peak_threshold = hourly_occupancy.quantile(0.75)
        peak_hours = hourly_occupancy[hourly_occupancy >= peak_threshold].index.tolist()
        
        # Group consecutive peak hours
        peak_periods = []
        if peak_hours:
            current_period = [peak_hours[0]]
            
            for hour in peak_hours[1:]:
                if hour == current_period[-1] + 1:
                    current_period.append(hour)
                else:
                    if len(current_period) > 0:
                        peak_periods.append({
                            'start_hour': current_period[0],
                            'end_hour': current_period[-1],
                            'duration_hours': len(current_period),
                            'avg_occupancy': hourly_occupancy[current_period].mean()
                        })
                    current_period = [hour]
            
            # Add the last period
            if len(current_period) > 0:
                peak_periods.append({
                    'start_hour': current_period[0],
                    'end_hour': current_period[-1],
                    'duration_hours': len(current_period),
                    'avg_occupancy': hourly_occupancy[current_period].mean()
                })
        
        return {
            'peak_threshold': peak_threshold,
            'peak_periods': peak_periods,
            'total_peak_hours': len(peak_hours)
        }
    
    def _calculate_service_rate(self, df: pd.DataFrame) -> float:
        """Calculate service rate (vehicles processed per unit time)."""
        try:
            if 'vehicle_count' not in df.columns or 'timestamp' not in df.columns:
                return 0.0
            
            # Calculate time span
            time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            
            if time_span_hours <= 0:
                return 0.0
            
            # Service rate = vehicles processed per hour
            total_vehicles = df['vehicle_count'].sum()
            service_rate = total_vehicles / time_span_hours
            
            return service_rate
            
        except Exception:
            return 0.0
    
    def _calculate_queue_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate queue efficiency based on wait times and throughput."""
        try:
            if 'avg_wait_time_sec' not in df.columns or 'throughput_vph' not in df.columns:
                return 0.0
            
            avg_wait_time = df['avg_wait_time_sec'].mean()
            avg_throughput = df['throughput_vph'].mean()
            
            # Efficiency metric: higher throughput with lower wait times is better
            if avg_wait_time <= 0:
                return 100.0  # Perfect efficiency
            
            # Normalize efficiency score (0-100)
            ideal_wait_time = 30  # seconds (ideal target)
            efficiency = max(0, 100 - (avg_wait_time - ideal_wait_time) / ideal_wait_time * 50)
            
            return min(100, efficiency)
            
        except Exception:
            return 0.0
    
    def _calculate_speed_based_los(self, speeds: pd.Series) -> Dict[str, any]:
        """Calculate Level of Service based on speed."""
        avg_speed = speeds.mean()
        
        # Speed-based LOS thresholds (mph)
        if avg_speed >= 55:
            grade = 'A'
        elif avg_speed >= 45:
            grade = 'B'
        elif avg_speed >= 35:
            grade = 'C'
        elif avg_speed >= 25:
            grade = 'D'
        elif avg_speed >= 15:
            grade = 'E'
        else:
            grade = 'F'
        
        return {
            'grade': grade,
            'avg_speed': avg_speed,
            'description': self._get_los_description(grade)
        }
    
    def _calculate_density_based_los(self, densities: pd.Series) -> Dict[str, any]:
        """Calculate Level of Service based on traffic density."""
        avg_density = densities.mean()
        
        # Density-based LOS thresholds (vehicles per mile per lane)
        if avg_density <= 11:
            grade = 'A'
        elif avg_density <= 18:
            grade = 'B'
        elif avg_density <= 26:
            grade = 'C'
        elif avg_density <= 35:
            grade = 'D'
        elif avg_density <= 45:
            grade = 'E'
        else:
            grade = 'F'
        
        return {
            'grade': grade,
            'avg_density': avg_density,
            'description': self._get_los_description(grade)
        }
    
    def _calculate_vc_based_los(self, vc_ratio: float) -> Dict[str, any]:
        """Calculate Level of Service based on volume-to-capacity ratio."""
        # V/C ratio based LOS thresholds
        if vc_ratio <= 0.35:
            grade = 'A'
        elif vc_ratio <= 0.54:
            grade = 'B'
        elif vc_ratio <= 0.77:
            grade = 'C'
        elif vc_ratio <= 0.93:
            grade = 'D'
        elif vc_ratio <= 1.00:
            grade = 'E'
        else:
            grade = 'F'
        
        return {
            'grade': grade,
            'vc_ratio': vc_ratio,
            'description': self._get_los_description(grade)
        }
    
    def _get_los_description(self, grade: str) -> str:
        """Get description for Level of Service grade."""
        descriptions = {
            'A': 'Free flow - excellent conditions',
            'B': 'Stable flow - good conditions',
            'C': 'Stable flow - acceptable conditions',
            'D': 'Approaching unstable flow - tolerable conditions',
            'E': 'Unstable flow - poor conditions',
            'F': 'Forced flow - unacceptable conditions'
        }
        return descriptions.get(grade, 'Unknown')
    
    def _detect_speed_incidents(self, df: pd.DataFrame, sensitivity: float) -> List[Dict[str, any]]:
        """Detect incidents based on speed anomalies."""
        incidents = []
        
        try:
            speeds = df['avg_speed_mph']
            speed_mean = speeds.rolling(window=5, min_periods=1).mean()
            speed_std = speeds.rolling(window=5, min_periods=1).std()
            
            # Detect significant speed drops
            for i in range(len(df)):
                if i < 2:  # Skip first few points
                    continue
                
                current_speed = speeds.iloc[i]
                expected_speed = speed_mean.iloc[i-1]
                speed_threshold = expected_speed - (sensitivity * speed_std.iloc[i-1])
                
                if current_speed < speed_threshold and current_speed < expected_speed * 0.7:
                    incident = {
                        'type': 'Speed Drop',
                        'timestamp': df.iloc[i]['timestamp'],
                        'location': df.iloc[i].get('lane_id', 'Unknown'),
                        'severity': 'High' if current_speed < expected_speed * 0.5 else 'Medium',
                        'details': f"Speed dropped to {current_speed:.1f} mph (expected {expected_speed:.1f} mph)",
                        'duration_estimate': '15-30 minutes'
                    }
                    incidents.append(incident)
            
        except Exception as e:
            logger.warning(f"Error detecting speed incidents: {str(e)}")
        
        return incidents
    
    def _detect_volume_incidents(self, df: pd.DataFrame, sensitivity: float) -> List[Dict[str, any]]:
        """Detect incidents based on volume anomalies."""
        incidents = []
        
        try:
            if 'vehicle_count' not in df.columns:
                return incidents
            
            volumes = df['vehicle_count']
            volume_mean = volumes.rolling(window=5, min_periods=1).mean()
            volume_std = volumes.rolling(window=5, min_periods=1).std()
            
            for i in range(len(df)):
                if i < 2:
                    continue
                
                current_volume = volumes.iloc[i]
                expected_volume = volume_mean.iloc[i-1]
                
                # Detect significant volume drops (potential incident upstream)
                if current_volume < expected_volume - (sensitivity * volume_std.iloc[i-1]):
                    if current_volume < expected_volume * 0.6:
                        incident = {
                            'type': 'Volume Drop',
                            'timestamp': df.iloc[i]['timestamp'],
                            'location': df.iloc[i].get('lane_id', 'Unknown'),
                            'severity': 'Medium',
                            'details': f"Volume dropped to {current_volume} vehicles (expected {expected_volume:.1f})",
                            'duration_estimate': '20-45 minutes'
                        }
                        incidents.append(incident)
            
        except Exception as e:
            logger.warning(f"Error detecting volume incidents: {str(e)}")
        
        return incidents
    
    def _detect_queue_incidents(self, df: pd.DataFrame, sensitivity: float) -> List[Dict[str, any]]:
        """Detect incidents based on sudden queue length increases."""
        incidents = []
        
        try:
            if 'queue_length_vehicles' not in df.columns:
                return incidents
            
            queue_lengths = df['queue_length_vehicles']
            
            for i in range(1, len(df)):
                current_queue = queue_lengths.iloc[i]
                previous_queue = queue_lengths.iloc[i-1]
                
                # Detect sudden queue increases
                if current_queue > previous_queue + 10 and current_queue > previous_queue * 2:
                    incident = {
                        'type': 'Queue Buildup',
                        'timestamp': df.iloc[i]['timestamp'],
                        'location': df.iloc[i].get('lane_id', 'Unknown'),
                        'severity': 'High' if current_queue > 20 else 'Medium',
                        'details': f"Queue increased from {previous_queue} to {current_queue} vehicles",
                        'duration_estimate': '30-60 minutes'
                    }
                    incidents.append(incident)
            
        except Exception as e:
            logger.warning(f"Error detecting queue incidents: {str(e)}")
        
        return incidents
    
    def _deduplicate_incidents(self, incidents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Remove duplicate incident detections."""
        if not incidents:
            return incidents
        
        # Sort by timestamp
        incidents.sort(key=lambda x: x['timestamp'])
        
        deduplicated = []
        for incident in incidents:
            # Check if similar incident exists within 10 minutes
            is_duplicate = False
            for existing in deduplicated:
                time_diff = abs((incident['timestamp'] - existing['timestamp']).total_seconds())
                location_match = incident.get('location') == existing.get('location')
                type_match = incident.get('type') == existing.get('type')
                
                if time_diff < 600 and location_match and type_match:  # 10 minutes
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(incident)
        
        return deduplicated
    
    def _calculate_fuel_waste_cost(self, df: pd.DataFrame, fuel_cost_per_gallon: float) -> float:
        """Calculate fuel waste cost due to congestion."""
        try:
            if 'avg_speed_mph' not in df.columns or 'vehicle_count' not in df.columns:
                return 0.0
            
            # Fuel efficiency curves (simplified)
            def get_mpg(speed):
                # Optimal around 45-55 mph
                if speed <= 0:
                    return 0
                elif speed < 20:
                    return 15  # Heavy congestion
                elif speed < 35:
                    return 20  # Moderate congestion
                elif speed < 55:
                    return 25  # Good flow
                else:
                    return 22  # Highway speeds
            
            total_fuel_waste = 0
            optimal_mpg = 25  # MPG at optimal speed
            trip_distance = 10  # Assumed average trip distance
            
            for _, row in df.iterrows():
                current_speed = row['avg_speed_mph']
                vehicle_count = row['vehicle_count']
                
                current_mpg = get_mpg(current_speed)
                
                if current_mpg > 0:
                    # Fuel consumption at current conditions
                    current_fuel = (trip_distance / current_mpg) * vehicle_count
                    # Optimal fuel consumption
                    optimal_fuel = (trip_distance / optimal_mpg) * vehicle_count
                    # Additional fuel due to congestion
                    fuel_waste = max(0, current_fuel - optimal_fuel)
                    total_fuel_waste += fuel_waste
            
            return total_fuel_waste * fuel_cost_per_gallon
            
        except Exception:
            return 0.0
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate summary statistics for the report."""
        try:
            stats = {}
            
            if 'occupancy_percent' in df.columns:
                stats['occupancy'] = {
                    'mean': df['occupancy_percent'].mean(),
                    'median': df['occupancy_percent'].median(),
                    'max': df['occupancy_percent'].max(),
                    'std': df['occupancy_percent'].std()
                }
            
            if 'avg_speed_mph' in df.columns:
                stats['speed'] = {
                    'mean': df['avg_speed_mph'].mean(),
                    'median': df['avg_speed_mph'].median(),
                    'min': df['avg_speed_mph'].min(),
                    'std': df['avg_speed_mph'].std()
                }
            
            if 'vehicle_count' in df.columns:
                stats['volume'] = {
                    'total': df['vehicle_count'].sum(),
                    'mean_hourly': df['vehicle_count'].mean(),
                    'peak_hourly': df['vehicle_count'].max(),
                    'std': df['vehicle_count'].std()
                }
            
            return stats
            
        except Exception:
            return {}
    
    def _generate_congestion_recommendations(self, df: pd.DataFrame, 
                                           metrics: CongestionMetrics,
                                           los_analysis: Dict[str, any]) -> List[str]:
        """Generate actionable congestion management recommendations."""
        recommendations = []
        
        try:
            # Speed-based recommendations
            if metrics.avg_speed < 35:
                recommendations.append("Consider implementing dynamic speed limits to smooth traffic flow")
            
            # Occupancy-based recommendations
            if metrics.occupancy_percent > 80:
                recommendations.append("High occupancy detected - consider opening additional lanes or implementing ramp metering")
            
            # LOS-based recommendations
            overall_los = los_analysis.get('overall_los', {}).get('grade', 'F')
            if overall_los in ['E', 'F']:
                recommendations.append("Poor Level of Service - urgent capacity improvements needed")
            elif overall_los == 'D':
                recommendations.append("Approaching unstable conditions - monitor closely and prepare interventions")
            
            # Queue-based recommendations
            if metrics.queue_length > 10:
                recommendations.append("Significant queuing detected - optimize toll booth operations and consider ETC expansion")
            
            # Delay-based recommendations
            if metrics.delay_minutes > 10:
                recommendations.append("High delays detected - implement incident management protocols and traveler information systems")
            
            # Economic impact recommendations
            economic_impact = self.calculate_economic_impact(df)
            if economic_impact.get('total_impact', 0) > 50000:  # $50k threshold
                recommendations.append("High economic impact of congestion - prioritize capacity and efficiency improvements")
            
            # Time-based recommendations
            if 'timestamp' in df.columns:
                hourly_patterns = self._analyze_hourly_patterns(df)
                peak_hours = [hour for hour, data in hourly_patterns['hourly_data'].items() 
                             if data['occupancy_percent'] > 70]
                
                if len(peak_hours) > 4:
                    recommendations.append("Extended peak periods - consider flexible work arrangements and congestion pricing")
            
            # Default recommendation if none generated
            if not recommendations:
                if overall_los in ['A', 'B', 'C']:
                    recommendations.append("Traffic conditions are acceptable - continue monitoring and maintain current operations")
                else:
                    recommendations.append("Monitor traffic conditions closely and be prepared for interventions during peak periods")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
            recommendations.append("Unable to generate specific recommendations - manual analysis recommended")
        
        return recommendations


class TrafficFlowAnalyzer:
    """
    Specialized analyzer for traffic flow metrics and fundamental diagrams.
    """
    
    def __init__(self):
        self.datetime_utils = DateTimeUtils()
    
    def calculate_flow_metrics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate fundamental traffic flow metrics.
        
        Args:
            df: DataFrame with traffic flow data
            
        Returns:
            Dictionary with flow analysis results
        """
        try:
            flow_metrics = {}
            
            # Basic flow parameters
            if all(col in df.columns for col in ['throughput_vph', 'density_vpm', 'avg_speed_mph']):
                flow_metrics['fundamental_diagram'] = self._analyze_fundamental_diagram(df)
            
            # Capacity analysis
            if 'throughput_vph' in df.columns:
                flow_metrics['capacity_analysis'] = self._analyze_capacity(df['throughput_vph'])
            
            # Flow stability
            if 'avg_speed_mph' in df.columns:
                flow_metrics['stability_metrics'] = self._calculate_stability_metrics(df)
            
            logger.info("Traffic flow analysis completed")
            return flow_metrics
            
        except Exception as e:
            logger.error(f"Error calculating flow metrics: {str(e)}")
            return {}
    
    def _analyze_fundamental_diagram(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze the fundamental traffic flow diagram."""
        try:
            # Extract flow parameters
            flow = df['throughput_vph']
            density = df['density_vpm']
            speed = df['avg_speed_mph']
            
            # Find capacity conditions
            max_flow_idx = flow.idxmax()
            capacity = flow.loc[max_flow_idx]
            critical_density = density.loc[max_flow_idx]
            critical_speed = speed.loc[max_flow_idx]
            
            # Free flow conditions
            free_flow_speed = speed[density < density.quantile(0.1)].mean()
            
            # Jam density (theoretical)
            jam_density = critical_density * 2  # Simplified estimate
            
            return {
                'capacity_vph': capacity,
                'critical_density_vpm': critical_density,
                'critical_speed_mph': critical_speed,
                'free_flow_speed_mph': free_flow_speed,
                'jam_density_vpm': jam_density
            }
            
        except Exception:
            return {}
    
    def _analyze_capacity(self, flow_data: pd.Series) -> Dict[str, float]:
        """Analyze capacity characteristics."""
        try:
            max_flow = flow_data.max()
            avg_flow = flow_data.mean()
            flow_95th = flow_data.quantile(0.95)
            
            # Capacity utilization
            capacity_utilization = (avg_flow / max_flow) * 100 if max_flow > 0 else 0
            
            return {
                'max_observed_flow': max_flow,
                'average_flow': avg_flow,
                'flow_95th_percentile': flow_95th,
                'capacity_utilization_percent': capacity_utilization
            }
            
        except Exception:
            return {}
    
    def _calculate_stability_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate flow stability metrics."""
        try:
            stability = {}
            
            # Speed variance
            if 'avg_speed_mph' in df.columns:
                speed_cv = df['avg_speed_mph'].std() / df['avg_speed_mph'].mean()
                stability['speed_coefficient_variation'] = speed_cv
            
            # Flow variance
            if 'throughput_vph' in df.columns:
                flow_cv = df['throughput_vph'].std() / df['throughput_vph'].mean()
                stability['flow_coefficient_variation'] = flow_cv
            
            # Oscillation detection
            if 'avg_speed_mph' in df.columns and len(df) > 10:
                speed_diff = df['avg_speed_mph'].diff()
                oscillations = ((speed_diff > 0) != (speed_diff.shift(1) > 0)).sum()
                stability['speed_oscillation_count'] = oscillations
            
            return stability
            
        except Exception:
            return {}


class WaitTimeCalculator:
    """
    Specialized calculator for queue dynamics and wait time estimation.
    """
    
    def __init__(self):
        self.processor = DataProcessor()
    
    def calculate_queue_dynamics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate comprehensive queue dynamics.
        
        Args:
            df: DataFrame with queue and service data
            
        Returns:
            Dictionary with queue analysis
        """
        try:
            queue_dynamics = {}
            
            # Basic queue metrics
            if 'queue_length_vehicles' in df.columns:
                queue_dynamics['queue_statistics'] = {
                    'max_queue_length': df['queue_length_vehicles'].max(),
                    'avg_queue_length': df['queue_length_vehicles'].mean(),
                    'queue_length_std': df['queue_length_vehicles'].std()
                }
            
            # Wait time analysis
            if 'avg_wait_time_sec' in df.columns:
                queue_dynamics['wait_time_analysis'] = self._analyze_wait_times(df)
            
            # Service efficiency
            queue_dynamics['service_efficiency'] = self._calculate_service_efficiency(df)
            
            logger.info("Queue dynamics analysis completed")
            return queue_dynamics
            
        except Exception as e:
            logger.error(f"Error calculating queue dynamics: {str(e)}")
            return {}
    
    def _analyze_wait_times(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze wait time patterns."""
        try:
            wait_times = df['avg_wait_time_sec']
            
            analysis = {
                'mean_wait_time': wait_times.mean(),
                'median_wait_time': wait_times.median(),
                'max_wait_time': wait_times.max(),
                'wait_time_90th_percentile': wait_times.quantile(0.90),
                'wait_time_95th_percentile': wait_times.quantile(0.95),
                'wait_time_std': wait_times.std()
            }
            
            # Service level metrics
            analysis['percent_served_under_30sec'] = (wait_times <= 30).mean() * 100
            analysis['percent_served_under_60sec'] = (wait_times <= 60).mean() * 100
            
            return analysis
            
        except Exception:
            return {}
    
    def _calculate_service_efficiency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate service efficiency metrics."""
        try:
            efficiency = {}
            
            # Throughput efficiency
            if 'vehicle_count' in df.columns and 'processing_time_sec' in df.columns:
                total_service_time = df['processing_time_sec'].sum()
                total_vehicles = df['vehicle_count'].sum()
                
                if total_service_time > 0:
                    efficiency['vehicles_per_hour'] = (total_vehicles / total_service_time) * 3600
            
            # Queue clearing rate
            if 'queue_length_vehicles' in df.columns:
                queue_changes = df['queue_length_vehicles'].diff()
                clearing_rate = abs(queue_changes[queue_changes < 0]).mean()
                efficiency['avg_queue_clearing_rate'] = clearing_rate
            
            return efficiency
            
        except Exception:
            return {}
