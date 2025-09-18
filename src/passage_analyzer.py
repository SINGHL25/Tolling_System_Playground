"""
Passage Analyzer Module
======================

Analyzes vehicle passage data, IVDC (In-Vehicle Device Communication) logs,
and traffic patterns for toll road operations.

Classes:
--------
- PassageAnalyzer: Main class for passage data analysis
- IVDCProcessor: IVDC log processing and validation
- TrafficPatternAnalyzer: Traffic pattern recognition and analysis

Example Usage:
--------------
>>> analyzer = PassageAnalyzer()
>>> stats = analyzer.get_daily_statistics(passage_data)
>>> ivdc_metrics = analyzer.analyze_ivdc_performance(ivdc_logs)
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PassageMetrics:
    """Data class for passage analysis metrics."""
    total_vehicles: int
    avg_speed: float
    peak_hour: str
    congestion_level: str
    ivdc_success_rate: float
    violation_count: int
    revenue_estimate: float

@dataclass
class IVDCMetrics:
    """Data class for IVDC performance metrics."""
    success_rate: float
    avg_response_time: float
    error_count: int
    timeout_count: int
    data_quality_score: float

class PassageAnalyzer:
    """
    Main class for analyzing vehicle passage data and traffic patterns.
    
    Attributes:
        config (dict): Configuration parameters
        processor (DataProcessor): Data processing utilities
        validator (ValidationUtils): Data validation utilities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PassageAnalyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._get_default_config()
        self.processor = DataProcessor()
        self.validator = ValidationUtils()
        self.datetime_utils = DateTimeUtils()
        
        logger.info("PassageAnalyzer initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'speed_limits': {'min': 5, 'max': 120},
            'congestion_thresholds': {'light': 30, 'moderate': 60, 'heavy': 80},
            'ivdc_timeout_seconds': 10,
            'outlier_detection': True,
            'data_quality_threshold': 0.95
        }
    
    def analyze_passages(self, df: pd.DataFrame, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> PassageMetrics:
        """
        Perform comprehensive passage analysis.
        
        Args:
            df: DataFrame with passage data
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            
        Returns:
            PassageMetrics object with analysis results
        """
        try:
            # Validate input data
            if not self.validator.validate_passage_data(df):
                raise ValueError("Invalid passage data format")
            
            # Filter by date range if provided
            if start_date or end_date:
                df = self.processor.filter_by_date_range(df, start_date, end_date)
            
            # Basic statistics
            total_vehicles = len(df)
            avg_speed = df['speed_mph'].mean() if 'speed_mph' in df.columns else 0
            
            # Peak hour analysis
            if 'timestamp' in df.columns:
                hourly_counts = df.groupby(df['timestamp'].dt.hour).size()
                peak_hour = f"{hourly_counts.idxmax()}:00"
            else:
                peak_hour = "N/A"
            
            # Congestion level
            if 'occupancy_percent' in df.columns:
                avg_occupancy = df['occupancy_percent'].mean()
                congestion_level = self._classify_congestion(avg_occupancy)
            else:
                congestion_level = "Unknown"
            
            # IVDC success rate
            ivdc_success_rate = 0.0
            if 'ivdc_success' in df.columns:
                ivdc_success_rate = df['ivdc_success'].mean() * 100
            
            # Violations
            violation_count = 0
            if 'violation' in df.columns:
                violation_count = df['violation'].sum()
            
            # Revenue estimate
            revenue_estimate = self._estimate_revenue(df)
            
            metrics = PassageMetrics(
                total_vehicles=total_vehicles,
                avg_speed=avg_speed,
                peak_hour=peak_hour,
                congestion_level=congestion_level,
                ivdc_success_rate=ivdc_success_rate,
                violation_count=violation_count,
                revenue_estimate=revenue_estimate
            )
            
            logger.info(f"Analyzed {total_vehicles} vehicle passages")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing passages: {str(e)}")
            raise
    
    def get_daily_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily traffic statistics.
        
        Args:
            df: DataFrame with passage data
            
        Returns:
            DataFrame with daily statistics
        """
        try:
            if 'timestamp' not in df.columns:
                raise ValueError("Timestamp column required for daily statistics")
            
            daily_stats = df.groupby(df['timestamp'].dt.date).agg({
                'vehicle_id': 'count',
                'speed_mph': ['mean', 'std'] if 'speed_mph' in df.columns else 'count',
                'vehicle_type': lambda x: x.mode().iloc[0] if not x.empty else 'Car'
            }).round(2)
            
            # Flatten multi-level columns
            daily_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                 for col in daily_stats.columns]
            
            daily_stats = daily_stats.reset_index()
            daily_stats.columns = ['date', 'total_vehicles', 'avg_speed', 'speed_std', 'dominant_vehicle_type']
            
            logger.info(f"Generated daily statistics for {len(daily_stats)} days")
            return daily_stats
            
        except Exception as e:
            logger.error(f"Error calculating daily statistics: {str(e)}")
            raise
    
    def analyze_ivdc_performance(self, df: pd.DataFrame) -> IVDCMetrics:
        """
        Analyze IVDC (In-Vehicle Device Communication) performance.
        
        Args:
            df: DataFrame with IVDC log data
            
        Returns:
            IVDCMetrics object with performance metrics
        """
        try:
            if df.empty:
                return IVDCMetrics(0, 0, 0, 0, 0)
            
            # Success rate
            success_rate = 0.0
            if 'ivdc_success' in df.columns:
                success_rate = df['ivdc_success'].mean() * 100
            
            # Response time
            avg_response_time = 0.0
            if 'processing_time_ms' in df.columns:
                avg_response_time = df['processing_time_ms'].mean()
            
            # Error counts
            error_count = 0
            if 'error_code' in df.columns:
                error_count = df[df['error_code'].notna()].shape[0]
            
            # Timeout count
            timeout_count = 0
            if 'processing_time_ms' in df.columns:
                timeout_threshold = self.config['ivdc_timeout_seconds'] * 1000
                timeout_count = df[df['processing_time_ms'] > timeout_threshold].shape[0]
            
            # Data quality score
            data_quality_score = self._calculate_data_quality_score(df)
            
            metrics = IVDCMetrics(
                success_rate=success_rate,
                avg_response_time=avg_response_time,
                error_count=error_count,
                timeout_count=timeout_count,
                data_quality_score=data_quality_score
            )
            
            logger.info(f"IVDC analysis complete - Success rate: {success_rate:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing IVDC performance: {str(e)}")
            raise
    
    def detect_traffic_anomalies(self, df: pd.DataFrame, 
                                threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect traffic anomalies using statistical methods.
        
        Args:
            df: DataFrame with traffic data
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly indicators
        """
        try:
            if 'vehicle_count' not in df.columns:
                logger.warning("vehicle_count column not found, creating from data")
                df['vehicle_count'] = df.groupby(df['timestamp'].dt.floor('H')).cumcount() + 1
            
            # Calculate hourly traffic counts
            hourly_traffic = df.groupby(df['timestamp'].dt.floor('H')).size().reset_index(name='vehicle_count')
            
            # Z-score based anomaly detection
            mean_traffic = hourly_traffic['vehicle_count'].mean()
            std_traffic = hourly_traffic['vehicle_count'].std()
            
            hourly_traffic['z_score'] = (hourly_traffic['vehicle_count'] - mean_traffic) / std_traffic
            hourly_traffic['is_anomaly'] = abs(hourly_traffic['z_score']) > threshold
            hourly_traffic['anomaly_type'] = np.where(
                hourly_traffic['z_score'] > threshold, 'High Traffic',
                np.where(hourly_traffic['z_score'] < -threshold, 'Low Traffic', 'Normal')
            )
            
            anomaly_count = hourly_traffic['is_anomaly'].sum()
            logger.info(f"Detected {anomaly_count} traffic anomalies")
            
            return hourly_traffic
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise
    
    def analyze_vehicle_types(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze vehicle type distribution and patterns.
        
        Args:
            df: DataFrame with vehicle data
            
        Returns:
            Dictionary with vehicle type analysis results
        """
        try:
            if 'vehicle_type' not in df.columns:
                logger.warning("No vehicle_type column found")
                return {}
            
            # Vehicle type distribution
            type_distribution = df['vehicle_type'].value_counts()
            type_percentages = df['vehicle_type'].value_counts(normalize=True) * 100
            
            # Speed analysis by vehicle type
            speed_by_type = {}
            if 'speed_mph' in df.columns:
                speed_by_type = df.groupby('vehicle_type')['speed_mph'].agg([
                    'mean', 'std', 'min', 'max'
                ]).round(2).to_dict('index')
            
            # Peak hours by vehicle type
            peak_hours_by_type = {}
            if 'timestamp' in df.columns:
                for vtype in df['vehicle_type'].unique():
                    vtype_data = df[df['vehicle_type'] == vtype]
                    hourly_counts = vtype_data.groupby(vtype_data['timestamp'].dt.hour).size()
                    peak_hours_by_type[vtype] = hourly_counts.idxmax()
            
            results = {
                'type_distribution': type_distribution.to_dict(),
                'type_percentages': type_percentages.to_dict(),
                'speed_by_type': speed_by_type,
                'peak_hours_by_type': peak_hours_by_type,
                'total_types': len(type_distribution)
            }
            
            logger.info(f"Analyzed {len(type_distribution)} vehicle types")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing vehicle types: {str(e)}")
            raise
    
    def calculate_lane_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate efficiency metrics for each lane.
        
        Args:
            df: DataFrame with lane-specific data
            
        Returns:
            DataFrame with lane efficiency metrics
        """
        try:
            if 'lane_id' not in df.columns:
                logger.warning("No lane_id column found")
                return pd.DataFrame()
            
            lane_metrics = df.groupby('lane_id').agg({
                'vehicle_id': 'count',
                'speed_mph': ['mean', 'std'] if 'speed_mph' in df.columns else 'count',
                'processing_time_ms': 'mean' if 'processing_time_ms' in df.columns else 'count',
                'ivdc_success': 'mean' if 'ivdc_success' in df.columns else 'count'
            }).round(2)
            
            # Flatten columns
            lane_metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                  for col in lane_metrics.columns]
            
            # Calculate efficiency score (normalized combination of metrics)
            if 'speed_mph_mean' in lane_metrics.columns:
                lane_metrics['efficiency_score'] = (
                    (lane_metrics['speed_mph_mean'] / 60) * 0.4 +  # Speed component
                    (lane_metrics.get('ivdc_success_mean', 0.95)) * 0.3 +  # IVDC success component
                    (1 / (lane_metrics.get('processing_time_ms_mean', 100) / 100)) * 0.3  # Processing time component
                ) * 100
            
            lane_metrics = lane_metrics.reset_index()
            
            logger.info(f"Calculated efficiency for {len(lane_metrics)} lanes")
            return lane_metrics
            
        except Exception as e:
            logger.error(f"Error calculating lane efficiency: {str(e)}")
            raise
    
    def _classify_congestion(self, occupancy: float) -> str:
        """Classify congestion level based on occupancy percentage."""
        thresholds = self.config['congestion_thresholds']
        
        if occupancy < thresholds['light']:
            return "Free Flow"
        elif occupancy < thresholds['moderate']:
            return "Light"
        elif occupancy < thresholds['heavy']:
            return "Moderate"
        else:
            return "Heavy"
    
    def _estimate_revenue(self, df: pd.DataFrame) -> float:
        """Estimate revenue from passage data."""
        try:
            if 'vehicle_type' not in df.columns:
                return 0.0
            
            # Default toll rates
            toll_rates = {
                'Car': 3.50,
                'Truck': 12.00,
                'Bus': 8.50,
                'Motorcycle': 2.00,
                'Trailer': 15.00
            }
            
            revenue = 0.0
            for vtype, rate in toll_rates.items():
                count = len(df[df['vehicle_type'] == vtype])
                revenue += count * rate
            
            return revenue
            
        except Exception as e:
            logger.warning(f"Error estimating revenue: {str(e)}")
            return 0.0
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness and validity."""
        try:
            if df.empty:
                return 0.0
            
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            completeness_score = 1 - missing_ratio
            
            # Check for invalid speeds if available
            validity_score = 1.0
            if 'speed_mph' in df.columns:
                speed_limits = self.config['speed_limits']
                invalid_speeds = df[
                    (df['speed_mph'] < speed_limits['min']) | 
                    (df['speed_mph'] > speed_limits['max'])
                ].shape[0]
                validity_score = 1 - (invalid_speeds / len(df))
            
            # Combined score
            quality_score = (completeness_score * 0.6 + validity_score * 0.4) * 100
            
            return min(100, max(0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error calculating data quality score: {str(e)}")
            return 50.0  # Default neutral score


class IVDCProcessor:
    """
    Specialized processor for IVDC (In-Vehicle Device Communication) data.
    """
    
    def __init__(self):
        self.processor = DataProcessor()
        self.validator = ValidationUtils()
    
    def process_ivdc_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean IVDC log data.
        
        Args:
            df: Raw IVDC log DataFrame
            
        Returns:
            Processed IVDC DataFrame
        """
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Parse timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Clean response times
            if 'processing_time_ms' in df.columns:
                df['processing_time_ms'] = pd.to_numeric(df['processing_time_ms'], errors='coerce')
                # Cap at reasonable maximum (30 seconds)
                df['processing_time_ms'] = df['processing_time_ms'].clip(upper=30000)
            
            # Create success flag if not present
            if 'ivdc_success' not in df.columns and 'error_code' in df.columns:
                df['ivdc_success'] = df['error_code'].isna()
            
            logger.info(f"Processed {len(df)} IVDC log entries")
            return df
            
        except Exception as e:
            logger.error(f"Error processing IVDC logs: {str(e)}")
            raise
    
    def validate_ivdc_data(self, df: pd.DataFrame) -> bool:
        """
        Validate IVDC data format and content.
        
        Args:
            df: IVDC DataFrame to validate
            
        Returns:
            Boolean indicating if data is valid
        """
        required_columns = ['timestamp', 'device_id']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            logger.error("Missing required IVDC columns")
            return False
        
        # Check data types
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception:
            logger.error("Invalid timestamp format in IVDC data")
            return False
        
        logger.info("IVDC data validation passed")
        return True


class TrafficPatternAnalyzer:
    """
    Specialized analyzer for traffic pattern recognition and analysis.
    """
    
    def __init__(self):
        self.datetime_utils = DateTimeUtils()
    
    def identify_rush_hours(self, df: pd.DataFrame, 
                           threshold_percentile: float = 75) -> Dict[str, List[int]]:
        """
        Identify rush hour patterns from traffic data.
        
        Args:
            df: Traffic data DataFrame
            threshold_percentile: Percentile threshold for rush hour identification
            
        Returns:
            Dictionary with weekday and weekend rush hours
        """
        try:
            if 'timestamp' not in df.columns:
                raise ValueError("Timestamp column required")
            
            # Separate weekdays and weekends
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['timestamp'].dt.weekday >= 5
            
            # Calculate hourly traffic volumes
            weekday_traffic = df[~df['is_weekend']].groupby('hour').size()
            weekend_traffic = df[df['is_weekend']].groupby('hour').size()
            
            # Identify rush hours using threshold
            weekday_threshold = weekday_traffic.quantile(threshold_percentile / 100)
            weekend_threshold = weekend_traffic.quantile(threshold_percentile / 100)
            
            weekday_rush_hours = weekday_traffic[weekday_traffic >= weekday_threshold].index.tolist()
            weekend_rush_hours = weekend_traffic[weekend_traffic >= weekend_threshold].index.tolist()
            
            return {
                'weekday_rush_hours': weekday_rush_hours,
                'weekend_rush_hours': weekend_rush_hours,
                'weekday_threshold': weekday_threshold,
                'weekend_threshold': weekend_threshold
            }
            
        except Exception as e:
            logger.error(f"Error identifying rush hours: {str(e)}")
            raise
    
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze seasonal traffic patterns.
        
        Args:
            df: Traffic data DataFrame spanning multiple seasons
            
        Returns:
            Dictionary with seasonal analysis results
        """
        try:
            if 'timestamp' not in df.columns:
                raise ValueError("Timestamp column required")
            
            df['month'] = df['timestamp'].dt.month
            df['season'] = df['month'].map(self._get_season)
            
            # Traffic by season
            seasonal_traffic = df.groupby('season').size()
            
            # Average daily traffic by season
            daily_traffic = df.groupby([df['timestamp'].dt.date, 'season']).size().reset_index(name='daily_count')
            avg_daily_by_season = daily_traffic.groupby('season')['daily_count'].mean()
            
            # Peak months
            monthly_traffic = df.groupby('month').size()
            peak_month = monthly_traffic.idxmax()
            low_month = monthly_traffic.idxmin()
            
            return {
                'seasonal_traffic': seasonal_traffic.to_dict(),
                'avg_daily_by_season': avg_daily_by_season.to_dict(),
                'peak_month': peak_month,
                'low_month': low_month,
                'seasonal_variance': seasonal_traffic.std() / seasonal_traffic.mean()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {str(e)}")
            raise
    
    def _get_season(self, month: int) -> str:
        """Map month number to season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
