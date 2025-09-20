"""
Utilities Module
===============

Common utilities and helper functions for the tolling system library.

Classes:
--------
- DataProcessor: Data processing and transformation utilities
- ValidationUtils: Data validation and quality checks
- DateTimeUtils: Date and time manipulation utilities
- ConfigManager: Configuration management
- Logger: Logging utilities

Example Usage:
--------------
>>> from src.utils import DataProcessor, ValidationUtils
>>> processor = DataProcessor()
>>> validator = ValidationUtils()
>>> clean_data = processor.clean_dataframe(raw_data)
>>> is_valid = validator.validate_passage_data(clean_data)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import os
import warnings
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Utility class for data processing and transformation operations.
    """
    
    def __init__(self):
        self.datetime_utils = DateTimeUtils()
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       drop_duplicates: bool = True,
                       handle_missing: str = 'drop') -> pd.DataFrame:
        """
        Clean DataFrame by handling missing values and duplicates.
        
        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            handle_missing: How to handle missing values ('drop', 'fill', 'ignore')
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Handle missing values
            if handle_missing == 'drop':
                initial_rows = len(df_clean)
                df_clean = df_clean.dropna()
                rows_dropped = initial_rows - len(df_clean)
                if rows_dropped > 0:
                    logger.info(f"Dropped {rows_dropped} rows with missing values")
            
            elif handle_missing == 'fill':
                # Fill numeric columns with median, categorical with mode
                for column in df_clean.columns:
                    if df_clean[column].dtype in ['int64', 'float64']:
                        df_clean[column].fillna(df_clean[column].median(), inplace=True)
                    else:
                        mode_value = df_clean[column].mode()
                        if not mode_value.empty:
                            df_clean[column].fillna(mode_value[0], inplace=True)
            
            # Drop duplicates
            if drop_duplicates:
                initial_rows = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                duplicates_dropped = initial_rows - len(df_clean)
                if duplicates_dropped > 0:
                    logger.info(f"Dropped {duplicates_dropped} duplicate rows")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning DataFrame: {str(e)}")
            return df
    
    def filter_by_date_range(self, df: pd.DataFrame, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           date_column: str = 'timestamp') -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: Input DataFrame
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            date_column: Name of date/timestamp column
            
        Returns:
            Filtered DataFrame
        """
        try:
            if date_column not in df.columns:
                logger.warning(f"Date column '{date_column}' not found")
                return df
            
            # Ensure datetime format
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Apply date filters
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df[date_column] >= start_dt]
                logger.info(f"Filtered data from {start_date}")
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df[date_column] <= end_dt]
                logger.info(f"Filtered data until {end_date}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error filtering by date range: {str(e)}")
            return df
    
    def aggregate_by_time_period(self, df: pd.DataFrame,
                               time_column: str = 'timestamp',
                               period: str = 'H',
                               agg_functions: Dict[str, str] = None) -> pd.DataFrame:
        """
        Aggregate data by time periods.
        
        Args:
            df: Input DataFrame
            time_column: Name of timestamp column
            period: Aggregation period ('H', 'D', 'W', 'M')
            agg_functions: Dictionary of column -> aggregation function
            
        Returns:
            Aggregated DataFrame
        """
        try:
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found")
            
            # Default aggregation functions
            if agg_functions is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                agg_functions = {col: 'mean' for col in numeric_cols if col != time_column}
            
            # Ensure datetime format
            df[time_column] = pd.to_datetime(df[time_column])
            
            # Group by time period and aggregate
            df_agg = df.groupby(pd.Grouper(key=time_column, freq=period)).agg(agg_functions)
            
            # Reset index to make timestamp a column
            df_agg = df_agg.reset_index()
            
            logger.info(f"Aggregated data by {period} periods")
            return df_agg
            
        except Exception as e:
            logger.error(f"Error aggregating by time period: {str(e)}")
            return df
    
    def normalize_columns(self, df: pd.DataFrame, 
                         columns: List[str] = None,
                         method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numeric columns in DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of columns to normalize (None for all numeric)
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            DataFrame with normalized columns
        """
        try:
            df_norm = df.copy()
            
            if columns is None:
                columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
            
            for column in columns:
                if column not in df_norm.columns:
                    logger.warning(f"Column '{column}' not found for normalization")
                    continue
                
                if method == 'minmax':
                    min_val = df_norm[column].min()
                    max_val = df_norm[column].max()
                    if max_val != min_val:
                        df_norm[column] = (df_norm[column] - min_val) / (max_val - min_val)
                
                elif method == 'zscore':
                    mean_val = df_norm[column].mean()
                    std_val = df_norm[column].std()
                    if std_val != 0:
                        df_norm[column] = (df_norm[column] - mean_val) / std_val
            
            logger.info(f"Normalized {len(columns)} columns using {method} method")
            return df_norm
            
        except Exception as e:
            logger.error(f"Error normalizing columns: {str(e)}")
            return df
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in DataFrame columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (None for all numeric)
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier indicators
        """
        try:
            df_outliers = df.copy()
            
            if columns is None:
                columns = df_outliers.select_dtypes(include=[np.number]).columns.tolist()
            
            for column in columns:
                if column not in df_outliers.columns:
                    continue
                
                if method == 'iqr':
                    Q1 = df_outliers[column].quantile(0.25)
                    Q3 = df_outliers[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = (df_outliers[column] < lower_bound) | (df_outliers[column] > upper_bound)
                
                elif method == 'zscore':
                    z_scores = np.abs((df_outliers[column] - df_outliers[column].mean()) / df_outliers[column].std())
                    outliers = z_scores > threshold
                
                df_outliers[f'{column}_outlier'] = outliers
            
            total_outliers = df_outliers[[col for col in df_outliers.columns if col.endswith('_outlier')]].any(axis=1).sum()
            logger.info(f"Detected {total_outliers} rows with outliers using {method} method")
            
            return df_outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return df
    
    def create_time_features(self, df: pd.DataFrame, 
                           timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Create time-based features from timestamp column.
        
        Args:
            df: Input DataFrame
            timestamp_column: Name of timestamp column
            
        Returns:
            DataFrame with additional time features
        """
        try:
            df_time = df.copy()
            
            if timestamp_column not in df_time.columns:
                logger.warning(f"Timestamp column '{timestamp_column}' not found")
                return df_time
            
            # Ensure datetime format
            df_time[timestamp_column] = pd.to_datetime(df_time[timestamp_column])
            
            # Extract time components
            df_time['year'] = df_time[timestamp_column].dt.year
            df_time['month'] = df_time[timestamp_column].dt.month
            df_time['day'] = df_time[timestamp_column].dt.day
            df_time['hour'] = df_time[timestamp_column].dt.hour
            df_time['minute'] = df_time[timestamp_column].dt.minute
            df_time['weekday'] = df_time[timestamp_column].dt.weekday
            df_time['week_of_year'] = df_time[timestamp_column].dt.isocalendar().week
            df_time['day_of_year'] = df_time[timestamp_column].dt.dayofyear
            
            # Create categorical features
            df_time['is_weekend'] = df_time['weekday'].isin([5, 6])
            df_time['is_business_hour'] = df_time['hour'].between(9, 17)
            df_time['is_rush_hour'] = df_time['hour'].isin([7, 8, 9, 17, 18, 19])
            
            # Cyclical encoding
            df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
            df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
            df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_year'] / 365)
            df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_year'] / 365)
            
            logger.info("Created time-based features")
            return df_time
            
        except Exception as e:
            logger.error(f"Error creating time features: {str(e)}")
            return df


class ValidationUtils:
    """
    Utility class for data validation and quality checks.
    """
    
    def __init__(self):
        pass
    
    def validate_passage_data(self, df: pd.DataFrame) -> bool:
        """
        Validate vehicle passage data format and content.
        
        Args:
            df: DataFrame with passage data
            
        Returns:
            Boolean indicating if data is valid
        """
        try:
            # Check required columns
            required_columns = ['timestamp', 'vehicle_id']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception:
                    logger.error("Cannot convert timestamp column to datetime")
                    return False
            
            # Check for reasonable data ranges
            if 'speed_mph' in df.columns:
                invalid_speeds = df[(df['speed_mph'] < 0) | (df['speed_mph'] > 200)]
                if len(invalid_speeds) > 0:
                    logger.warning(f"Found {len(invalid_speeds)} records with invalid speeds")
            
            # Check for future timestamps
            future_timestamps = df[df['timestamp'] > datetime.now() + timedelta(hours=1)]
            if len(future_timestamps) > 0:
                logger.warning(f"Found {len(future_timestamps)} future timestamps")
            
            logger.info("Passage data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating passage data: {str(e)}")
            return False
    
    def validate_transaction_data(self, df: pd.DataFrame) -> bool:
        """
        Validate transaction data format and content.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Boolean indicating if data is valid
        """
        try:
            # Check required columns
            required_columns = ['timestamp', 'final_toll']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception:
                    logger.error("Cannot convert timestamp column to datetime")
                    return False
            
            # Validate toll amounts
            if not pd.api.types.is_numeric_dtype(df['final_toll']):
                logger.error("final_toll column must be numeric")
                return False
            
            negative_tolls = df[df['final_toll'] < 0]
            if len(negative_tolls) > 0:
                logger.warning(f"Found {len(negative_tolls)} records with negative toll amounts")
            
            excessive_tolls = df[df['final_toll'] > 1000]  # $1000 seems excessive
            if len(excessive_tolls) > 0:
                logger.warning(f"Found {len(excessive_tolls)} records with very high toll amounts")
            
            logger.info("Transaction data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating transaction data: {str(e)}")
            return False
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Comprehensive data quality assessment.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            quality_report = {}
            
            # Basic info
            quality_report['total_rows'] = len(df)
            quality_report['total_columns'] = len(df.columns)
            
            # Missing values
            missing_values = df.isnull().sum()
            quality_report['missing_values'] = missing_values.to_dict()
            quality_report['missing_percentage'] = (missing_values / len(df) * 100).to_dict()
            
            # Duplicates
            duplicate_rows = df.duplicated().sum()
            quality_report['duplicate_rows'] = duplicate_rows
            quality_report['duplicate_percentage'] = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0
            
            # Data types
            quality_report['data_types'] = df.dtypes.astype(str).to_dict()
            
            # Numeric column statistics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_stats = {}
            for col in numeric_columns:
                numeric_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'zeros': (df[col] == 0).sum(),
                    'negatives': (df[col] < 0).sum()
                }
            quality_report['numeric_statistics'] = numeric_stats
            
            # Categorical column statistics
            categorical_columns = df.select_dtypes(include=['object']).columns
            categorical_stats = {}
            for col in categorical_columns:
                unique_values = df[col].nunique()
                categorical_stats[col] = {
                    'unique_values': unique_values,
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'cardinality_ratio': unique_values / len(df) if len(df) > 0 else 0
                }
            quality_report['categorical_statistics'] = categorical_stats
            
            # Overall quality score (0-100)
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) if len(df) > 0 else 0
            uniqueness = (1 - duplicate_rows / len(df)) if len(df) > 0 else 1
            quality_score = (completeness + uniqueness) / 2 * 100
            quality_report['overall_quality_score'] = quality_score
            
            logger.info(f"Data quality assessment complete - Score: {quality_score:.1f}/100")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return {}
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, str]) -> bool:
        """
        Validate DataFrame against expected schema.
        
        Args:
            df: DataFrame to validate
            expected_schema: Dictionary of column_name -> expected_dtype
            
        Returns:
            Boolean indicating if schema is valid
        """
        try:
            schema_issues = []
            
            # Check for missing columns
            missing_columns = set(expected_schema.keys()) - set(df.columns)
            if missing_columns:
                schema_issues.append(f"Missing columns: {missing_columns}")
            
            # Check data types
            for column, expected_dtype in expected_schema.items():
                if column in df.columns:
                    actual_dtype = str(df[column].dtype)
                    
                    # Flexible type checking
                    if expected_dtype in ['int64', 'int32', 'int'] and not pd.api.types.is_integer_dtype(df[column]):
                        schema_issues.append(f"Column '{column}' expected {expected_dtype}, got {actual_dtype}")
                    elif expected_dtype in ['float64', 'float32', 'float'] and not pd.api.types.is_numeric_dtype(df[column]):
                        schema_issues.append(f"Column '{column}' expected {expected_dtype}, got {actual_dtype}")
                    elif expected_dtype == 'datetime64[ns]' and not pd.api.types.is_datetime64_any_dtype(df[column]):
                        schema_issues.append(f"Column '{column}' expected {expected_dtype}, got {actual_dtype}")
            
            if schema_issues:
                for issue in schema_issues:
                    logger.warning(issue)
                return False
            
            logger.info("Schema validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            return False


class DateTimeUtils:
    """
    Utility class for date and time manipulation operations.
    """
    
    def __init__(self):
        pass
    
    def parse_datetime(self, date_string: str, format_string: str = None) -> datetime:
        """
        Parse date string to datetime object.
        
        Args:
            date_string: String representation of date/time
            format_string: Expected format (None for auto-detection)
            
        Returns:
            Datetime object
        """
        try:
            if format_string:
                return datetime.strptime(date_string, format_string)
            else:
                return pd.to_datetime(date_string)
        except Exception as e:
            logger.error(f"Error parsing datetime '{date_string}': {str(e)}")
            raise
    
    def get_business_hours(self, start_time: str = "09:00", 
                          end_time: str = "17:00") -> Tuple[int, int]:
        """
        Get business hours as hour integers.
        
        Args:
            start_time: Business start time (HH:MM)
            end_time: Business end time (HH:MM)
            
        Returns:
            Tuple of (start_hour, end_hour)
        """
        try:
            start_hour = int(start_time.split(':')[0])
            end_hour = int(end_time.split(':')[0])
            return start_hour, end_hour
        except Exception as e:
            logger.error(f"Error parsing business hours: {str(e)}")
            return 9, 17  # Default
    
    def is_weekend(self, date: datetime) -> bool:
        """Check if date falls on weekend."""
        return date.weekday() >= 5
    
    def is_holiday(self, date: datetime, holidays: List[str] = None) -> bool:
        """
        Check if date is a holiday.
        
        Args:
            date: Date to check
            holidays: List of holiday dates in YYYY-MM-DD format
            
        Returns:
            Boolean indicating if date is a holiday
        """
        try:
            if holidays is None:
                # Default US federal holidays (simplified)
                holidays = [
                    f"{date.year}-01-01",  # New Year's Day
                    f"{date.year}-07-04",  # Independence Day
                    f"{date.year}-12-25",  # Christmas Day
                ]
            
            date_str = date.strftime('%Y-%m-%d')
            return date_str in holidays
            
        except Exception:
            return False
    
    def get_time_periods(self, df: pd.DataFrame, 
                        timestamp_column: str = 'timestamp') -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame into different time periods.
        
        Args:
            df: Input DataFrame
            timestamp_column: Name of timestamp column
            
        Returns:
            Dictionary of time period DataFrames
        """
        try:
            if timestamp_column not in df.columns:
                return {}
            
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            
            periods = {}
            
            # Split by weekday/weekend
            periods['weekday'] = df[df[timestamp_column].dt.weekday < 5]
            periods['weekend'] = df[df[timestamp_column].dt.weekday >= 5]
            
            # Split by business hours
            business_hours = df[df[timestamp_column].dt.hour.between(9, 17)]
            periods['business_hours'] = business_hours
            periods['non_business_hours'] = df[~df.index.isin(business_hours.index)]
            
            # Split by rush hours
            morning_rush = df[df[timestamp_column].dt.hour.between(7, 9)]
            evening_rush = df[df[timestamp_column].dt.hour.between(17, 19)]
            periods['morning_rush'] = morning_rush
            periods['evening_rush'] = evening_rush
            periods['off_peak'] = df[~df.index.isin(morning_rush.index.union(evening_rush.index))]
            
            return periods
            
        except Exception as e:
            logger.error(f"Error splitting time periods: {str(e)}")
            return {}


class ConfigManager:
    """
    Utility class for configuration management.
    """
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_file}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def save_config(self, config_file: str = None) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config_file: Path to save configuration (optional)
            
        Returns:
            Boolean indicating success
        """
        try:
            file_path = config_file or self.config_file
            if not file_path:
                raise ValueError("No config file specified")
            
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: any = None) -> any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, new_config: Dict[str, any]) -> None:
        """Update configuration with new values."""
        self.config.update(new_config)


class Logger:
    """
    Enhanced logging utility with custom formatting and handlers.
    """
    
    def __init__(self, name: str = __name__, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup console and file handlers with custom formatting."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler (optional)
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'tolling_system.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Custom formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)


class SecurityUtils:
    """
    Utility class for security-related operations.
    """
    
    @staticmethod
    def hash_data(data: str, algorithm: str = 'sha256') -> str:
        """
        Hash data using specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hashing algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            Hexadecimal hash string
        """
        try:
            hash_func = getattr(hashlib, algorithm)()
            hash_func.update(data.encode('utf-8'))
            return hash_func.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing data: {str(e)}")
            return ""
    
    @staticmethod
    def anonymize_data(df: pd.DataFrame, 
                      columns: List[str],
                      method: str = 'hash') -> pd.DataFrame:
        """
        Anonymize sensitive data in DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of columns to anonymize
            method: Anonymization method ('hash', 'mask', 'remove')
            
        Returns:
            DataFrame with anonymized data
        """
        try:
            df_anon = df.copy()
            
            for column in columns:
                if column not in df_anon.columns:
                    continue
                
                if method == 'hash':
                    df_anon[column] = df_anon[column].astype(str).apply(
                        lambda x: SecurityUtils.hash_data(x)[:8]  # First 8 chars of hash
                    )
                elif method == 'mask':
                    df_anon[column] = df_anon[column].astype(str).apply(
                        lambda x: '*' * len(x) if len(x) > 0 else x
                    )
                elif method == 'remove':
                    df_anon = df_anon.drop(columns=[column])
            
            logger.info(f"Anonymized {len(columns)} columns using {method} method")
            return df_anon
            
        except Exception as e:
            logger.error(f"Error anonymizing data: {str(e)}")
            return df


class PerformanceUtils:
    """
    Utility class for performance monitoring and optimization.
    """
    
    @staticmethod
    def time_function(func):
        """Decorator to time function execution."""
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def memory_usage(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate memory usage of DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with memory usage statistics
        """
        try:
            memory_usage = df.memory_usage(deep=True)
            total_mb = memory_usage.sum() / (1024 * 1024)
            
            return {
                'total_mb': total_mb,
                'per_column_mb': (memory_usage / (1024 * 1024)).to_dict(),
                'rows': len(df),
                'columns': len(df.columns),
                'mb_per_row': total_mb / len(df) if len(df) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating memory usage: {str(e)}")
            return {}
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        try:
            df_optimized = df.copy()
            
            # Optimize integer columns
            int_columns = df_optimized.select_dtypes(include=['int64']).columns
            for col in int_columns:
                col_min = df_optimized[col].min()
                col_max = df_optimized[col].max()
                
                if col_min >= 0:  # Unsigned integers
                    if col_max < 255:
                        df_optimized[col] = df_optimized[col].astype('uint8')
                    elif col_max < 65535:
                        df_optimized[col] = df_optimized[col].astype('uint16')
                    elif col_max < 4294967295:
                        df_optimized[col] = df_optimized[col].astype('uint32')
                else:  # Signed integers
                    if col_min > -128 and col_max < 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df_optimized[col] = df_optimized[col].astype('int32')
            
            # Optimize float columns
            float_columns = df_optimized.select_dtypes(include=['float64']).columns
            for col in float_columns:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            
            # Convert object columns to category if beneficial
            object_columns = df_optimized.select_dtypes(include=['object']).columns
            for col in object_columns:
                num_unique_values = df_optimized[col].nunique()
                num_total_values = len(df_optimized[col])
                
                # Convert to category if cardinality is less than 50% of total values
                if num_unique_values / num_total_values < 0.5:
                    df_optimized[col] = df_optimized[col].astype('category')
            
            # Calculate memory savings
            original_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            optimized_mb = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
            savings_percent = ((original_mb - optimized_mb) / original_mb) * 100 if original_mb > 0 else 0
            
            logger.info(f"DataFrame optimized - Memory reduced by {savings_percent:.1f}% ({original_mb:.1f}MB -> {optimized_mb:.1f}MB)")
            
            return df_optimized
            
        except Exception as e:
            logger.error(f"Error optimizing DataFrame: {str(e)}")
            return df
