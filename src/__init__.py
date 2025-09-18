"""
Tolling System Core Library
==========================

A comprehensive library for toll road data analysis, traffic management,
and revenue optimization.

Modules:
--------
- passage_analyzer: Vehicle passage data processing and IVDC log analysis
- transaction_analyzer: Revenue analytics and transaction processing
- congestion_estimator: Traffic congestion KPIs and wait time calculations
- pollution_analyzer: Air quality monitoring and emissions analysis
- health_monitor: System health monitoring and device status tracking
- stats_visualizer: Statistical analysis and visualization utilities
- ml_models: Machine learning models for traffic and revenue prediction
- utils: Common utilities and helper functions

Example Usage:
--------------
>>> from src.passage_analyzer import PassageAnalyzer
>>> from src.transaction_analyzer import TransactionAnalyzer
>>> from src.ml_models import TrafficPredictor

>>> # Analyze vehicle passages
>>> passage_analyzer = PassageAnalyzer()
>>> daily_stats = passage_analyzer.get_daily_statistics(data)

>>> # Analyze transactions
>>> transaction_analyzer = TransactionAnalyzer()
>>> revenue_metrics = transaction_analyzer.calculate_revenue_metrics(transactions)

>>> # Predict traffic
>>> predictor = TrafficPredictor()
>>> predictions = predictor.predict_traffic(features)
"""

__version__ = "1.0.0"
__author__ = "Tolling System Development Team"
__email__ = "dev@tollingsystem.com"

# Import main classes for easy access
from .passage_analyzer import PassageAnalyzer
from .transaction_analyzer import TransactionAnalyzer
from .congestion_estimator import CongestionEstimator
from .pollution_analyzer import PollutionAnalyzer
from .health_monitor import HealthMonitor
from .stats_visualizer import StatsVisualizer
from .ml_models import (
    TrafficPredictor, 
    RevenuePredictor, 
    AnomalyDetector,
    ModelTrainer
)
from .utils import (
    DataProcessor,
    ConfigManager,
    Logger,
    ValidationUtils,
    DateTimeUtils
)

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

# Configuration defaults
DEFAULT_CONFIG = {
    'data_retention_days': 365,
    'sampling_rate_minutes': 15,
    'prediction_horizon_hours': 72,
    'alert_thresholds': {
        'traffic_high': 80,
        'system_downtime': 95,
        'revenue_low': 50,
        'air_quality_poor': 100
    },
    'database_config': {
        'host': 'localhost',
        'port': 5432,
        'name': 'tolling_system',
        'pool_size': 10
    }
}

# Supported data formats
SUPPORTED_FORMATS = ['csv', 'parquet', 'json', 'xlsx']

# System constants
SYSTEM_CONSTANTS = {
    'MAX_VEHICLE_SPEED': 120,  # mph
    'MIN_VEHICLE_SPEED': 5,    # mph
    'LANE_CAPACITY_HOURLY': 1800,  # vehicles per hour
    'TOLL_RATES': {
        'Car': 3.50,
        'Truck': 12.00,
        'Bus': 8.50,
        'Motorcycle': 2.00,
        'Trailer': 15.00
    },
    'AQI_BREAKPOINTS': {
        'PM25': [(0, 12), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4)],
        'NO2': [(0, 53), (54, 100), (101, 360), (361, 649), (650, 1249)]
    }
}

def get_version():
    """Get the current version string."""
    return __version__

def get_system_info():
    """Get comprehensive system information."""
    return {
        'version': __version__,
        'version_info': VERSION_INFO,
        'supported_formats': SUPPORTED_FORMATS,
        'default_config': DEFAULT_CONFIG,
        'constants': SYSTEM_CONSTANTS
    }

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
