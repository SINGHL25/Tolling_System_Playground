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
            df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] /
