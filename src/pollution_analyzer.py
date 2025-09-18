"""
Pollution Analyzer Module
=========================

Analyzes air quality data, emissions impact, and environmental metrics
for toll road operations and sustainability monitoring.

Classes:
--------
- PollutionAnalyzer: Main class for air quality analysis
- EmissionsCalculator: Vehicle emissions calculations
- AQICalculator: Air Quality Index calculations and interpretations

Example Usage:
--------------
>>> analyzer = PollutionAnalyzer()
>>> aqi_metrics = analyzer.calculate_aqi_metrics(air_quality_data)
>>> emissions = analyzer.estimate_vehicle_emissions(traffic_data)
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
class AQIMetrics:
    """Data class for Air Quality Index metrics."""
    overall_aqi: float
    aqi_category: str
    health_risk: str
    pm25_aqi: float
    no2_aqi: float
    dominant_pollutant: str
    health_advisory: str

@dataclass
class EmissionMetrics:
    """Data class for vehicle emission metrics."""
    co2_emissions_kg: float
    nox_emissions_kg: float
    pm_emissions_kg: float
    fuel_consumption_gallons: float
    emission_rate_per_vehicle: float

class PollutionAnalyzer:
    """
    Main class for analyzing air quality data and environmental impact.
    
    Attributes:
        config (dict): Configuration parameters
        processor (DataProcessor): Data processing utilities
        validator (ValidationUtils): Data validation utilities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize PollutionAnalyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._get_default_config()
        self.processor = DataProcessor()
        self.validator = ValidationUtils()
        self.datetime_utils = DateTimeUtils()
        
        # Initialize calculators
        self.emissions_calculator = EmissionsCalculator()
        self.aqi_calculator = AQICalculator()
        
        logger.info("PollutionAnalyzer initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'aqi_breakpoints': {
                'PM25': [(0, 12.0), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4)],
                'NO2': [(0, 53), (54, 100), (101, 360), (361, 649), (650, 1249)],
                'O3': [(0, 54), (55, 70), (71, 85), (86, 105), (106, 200)],
                'CO': [(0, 4.4), (4.5, 9.4), (9.5, 12.4), (12.5, 15.4), (15.5, 30.4)]
            },
            'health_thresholds': {
                'good': 50,
                'moderate': 100,
                'unhealthy_sensitive': 150,
                'unhealthy': 200,
                'very_unhealthy': 300
            },
            'emission_factors': {
                'car': {'co2': 404, 'nox': 0.4, 'pm': 0.01},     # grams per mile
                'truck': {'co2': 1690, 'nox': 4.5, 'pm': 0.1},
                'bus': {'co2': 1325, 'nox': 11.0, 'pm': 0.3},
                'motorcycle': {'co2': 180, 'nox': 0.2, 'pm': 0.005}
            },
            'weather_adjustments': {
                'rain': 0.7,    # Rain washes out pollutants
                'wind': 0.5,    # Wind disperses pollutants
