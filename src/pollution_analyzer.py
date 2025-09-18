'weather_adjustments': {
                'rain': 0.7,    # Rain washes out pollutants
                'wind': 0.5,    # Wind disperses pollutants
                'fog': 1.3,     # Fog traps pollutants
                'clear': 1.0    # Baseline conditions
            },
            'monitoring_stations': {
                'max_distance_km': 10,
                'data_validity_hours': 2
            }
        }
    
    def calculate_aqi_metrics(self, df: pd.DataFrame) -> AQIMetrics:
        """
        Calculate comprehensive Air Quality Index metrics.
        
        Args:
            df: DataFrame with pollutant concentration data
            
        Returns:
            AQIMetrics object with AQI analysis results
        """
        try:
            if df.empty:
                logger.warning("No air quality data available")
                return self._empty_aqi_metrics()
            
            # Calculate individual pollutant AQIs
            pollutant_aqis = {}
            
            if 'pm25_ugm3' in df.columns:
                pollutant_aqis['PM25'] = self.aqi_calculator.calculate_pm25_aqi(df['pm25_ugm3'].mean())
            
            if 'no2_ugm3' in df.columns:
                pollutant_aqis['NO2'] = self.aqi_calculator.calculate_no2_aqi(df['no2_ugm3'].mean())
            
            if 'o3_ugm3' in df.columns:
                pollutant_aqis['O3'] = self.aqi_calculator.calculate_o3_aqi(df['o3_ugm3'].mean())
            
            if 'co_mgm3' in df.columns:
                pollutant_aqis['CO'] = self.aqi_calculator.calculate_co_aqi(df['co_mgm3'].mean())
            
            if not pollutant_aqis:
                logger.warning("No valid pollutant data for AQI calculation")
                return self._empty_aqi_metrics()
            
            # Overall AQI is the maximum of individual pollutant AQIs
            overall_aqi = max(pollutant_aqis.values())
            dominant_pollutant = max(pollutant_aqis, key=pollutant_aqis.get)
            
            # Determine AQI category and health risk
            aqi_category, health_risk = self._classify_aqi(overall_aqi)
            health_advisory = self._get_health_advisory(overall_aqi, aqi_category)
            
            metrics = AQIMetrics(
                overall_aqi=overall_aqi,
                aqi_category=aqi_category,
                health_risk=health_risk,
                pm25_aqi=pollutant_aqis.get('PM25', 0),
                no2_aqi=pollutant_aqis.get('NO2', 0),
                dominant_pollutant=dominant_pollutant,
                health_advisory=health_advisory
            )
            
            logger.info(f"AQI calculated: {overall_aqi:.0f} ({aqi_category})")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating AQI metrics: {str(e)}")
            return self._empty_aqi_metrics()
    
    def estimate_vehicle_emissions(self, df: pd.DataFrame) -> EmissionMetrics:
        """
        Estimate vehicle emissions from traffic data.
        
        Args:
            df: DataFrame with traffic data including vehicle counts and types
            
        Returns:
            EmissionMetrics object with emission estimates
        """
        try:
            if df.empty or 'vehicle_type' not in df.columns:
                logger.warning("No vehicle data available for emission estimation")
                return EmissionMetrics(0, 0, 0, 0, 0)
            
            total_co2 = 0
            total_nox = 0
            total_pm = 0
            total_fuel = 0
            total_vehicles = 0
            
            # Calculate emissions by vehicle type
            for vehicle_type in df['vehicle_type'].unique():
                vehicle_data = df[df['vehicle_type'] == vehicle_type]
                vehicle_count = len(vehicle_data)
                
                if vehicle_count == 0:
                    continue
                
                # Get emission factors for this vehicle type
                emission_factor = self.config['emission_factors'].get(
                    vehicle_type.lower(), 
                    self.config['emission_factors']['car']  # Default to car
                )
                
                # Estimate distance traveled (assume average trip through toll road)
                avg_distance_miles = 10  # Configurable assumption
                
                # Calculate total emissions for this vehicle type
                type_co2 = vehicle_count * emission_factor['co2'] * avg_distance_miles / 1000  # Convert g to kg
                type_nox = vehicle_count * emission_factor['nox'] * avg_distance_miles / 1000
                type_pm = vehicle_count * emission_factor['pm'] * avg_distance_miles / 1000
                
                # Fuel consumption (rough estimate based on CO2)
                type_fuel = type_co2 * 0.43  # kg CO2 to gallons (approximate conversion)
                
                total_co2 += type_co2
                total_nox += type_nox
                total_pm += type_pm
                total_fuel += type_fuel
                total_vehicles += vehicle_count
            
            # Calculate per-vehicle emission rate
            emission_rate = (total_co2 + total_nox + total_pm) / total_vehicles if total_vehicles > 0 else 0
            
            metrics = EmissionMetrics(
                co2_emissions_kg=total_co2,
                nox_emissions_kg=total_nox,
                pm_emissions_kg=total_pm,
                fuel_consumption_gallons=total_fuel,
                emission_rate_per_vehicle=emission_rate
            )
            
            logger.info(f"Emissions estimated: {total_co2:.1f} kg CO2, {total_nox:.2f} kg NOx")
            return metrics
            
        except Exception as e:
            logger.error(f"Error estimating vehicle emissions: {str(e)}")
            return EmissionMetrics(0, 0, 0, 0, 0)
    
    def analyze_pollution_trends(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze pollution trends over time and by conditions.
        
        Args:
            df: DataFrame with pollution data over time
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            if df.empty or 'timestamp' not in df.columns:
                return {}
            
            trends = {}
            
            # Temporal trends
            trends['temporal_trends'] = self._analyze_temporal_trends(df)
            
            # Weather correlation
            if 'weather_condition' in df.columns:
                trends['weather_correlation'] = self._analyze_weather_correlation(df)
            
            # Traffic correlation
            if 'vehicle_count_estimated' in df.columns:
                trends['traffic_correlation'] = self._analyze_traffic_correlation(df)
            
            # Spatial analysis (if multiple stations)
            if 'station_id' in df.columns:
                trends['spatial_analysis'] = self._analyze_spatial_patterns(df)
            
            # Health impact trends
            trends['health_impact'] = self._analyze_health_impact_trends(df)
            
            logger.info("Pollution trend analysis completed")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing pollution trends: {str(e)}")
            return {}
    
    def calculate_health_costs(self, df: pd.DataFrame, 
                             population: int = 50000) -> Dict[str, float]:
        """
        Calculate health costs associated with air pollution.
        
        Args:
            df: DataFrame with pollution data
            population: Affected population size
            
        Returns:
            Dictionary with health cost estimates
        """
        try:
            if df.empty:
                return {}
            
            health_costs = {}
            
            # PM2.5 health costs
            if 'pm25_ugm3' in df.columns:
                pm25_costs = self._calculate_pm25_health_costs(df['pm25_ugm3'], population)
                health_costs.update(pm25_costs)
            
            # NO2 health costs
            if 'no2_ugm3' in df.columns:
                no2_costs = self._calculate_no2_health_costs(df['no2_ugm3'], population)
                health_costs.update(no2_costs)
            
            # Total health costs
            cost_components = ['pm25_mortality_cost', 'pm25_morbidity_cost', 
                             'no2_respiratory_cost', 'no2_cardiovascular_cost']
            total_cost = sum(health_costs.get(component, 0) for component in cost_components)
            health_costs['total_annual_health_cost'] = total_cost
            
            # Per capita costs
            if population > 0:
                health_costs['annual_cost_per_capita'] = total_cost / population
            
            logger.info(f"Health costs calculated: ${total_cost:,.2f} annually")
            return health_costs
            
        except Exception as e:
            logger.error(f"Error calculating health costs: {str(e)}")
            return {}
    
    def generate_pollution_report(self, df: pd.DataFrame,
                                period_name: str = "Analysis Period") -> Dict[str, any]:
        """
        Generate comprehensive pollution analysis report.
        
        Args:
            df: DataFrame with pollution data
            period_name: Name of the analysis period
            
        Returns:
            Dictionary with complete pollution analysis
        """
        try:
            if df.empty:
                logger.warning("No data available for pollution report")
                return {}
            
            # Core analyses
            aqi_metrics = self.calculate_aqi_metrics(df)
            
            # Vehicle emissions (if traffic data available)
            emission_metrics = EmissionMetrics(0, 0, 0, 0, 0)
            if 'vehicle_type' in df.columns:
                emission_metrics = self.estimate_vehicle_emissions(df)
            
            pollution_trends = self.analyze_pollution_trends(df)
            health_costs = self.calculate_health_costs(df)
            
            # Summary statistics
            summary_stats = self._calculate_pollution_statistics(df)
            
            # Recommendations
            recommendations = self._generate_pollution_recommendations(df, aqi_metrics)
            
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
                'aqi_metrics': aqi_metrics.__dict__,
                'emission_metrics': emission_metrics.__dict__,
                'pollution_trends': pollution_trends,
                'health_costs': health_costs,
                'summary_statistics': summary_stats,
                'recommendations': recommendations
            }
            
            logger.info(f"Pollution report generated for {period_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating pollution report: {str(e)}")
            return {}
    
    # Helper methods
    
    def _empty_aqi_metrics(self) -> AQIMetrics:
        """Return empty AQI metrics."""
        return AQIMetrics(
            overall_aqi=0,
            aqi_category="Unknown",
            health_risk="Unknown",
            pm25_aqi=0,
            no2_aqi=0,
            dominant_pollutant="Unknown",
            health_advisory="No data available"
        )
    
    def _classify_aqi(self, aqi_value: float) -> Tuple[str, str]:
        """Classify AQI value into category and health risk."""
        if aqi_value <= 50:
            return "Good", "Low"
        elif aqi_value <= 100:
            return "Moderate", "Low"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups", "Medium"
        elif aqi_value <= 200:
            return "Unhealthy", "High"
        elif aqi_value <= 300:
            return "Very Unhealthy", "Very High"
        else:
            return "Hazardous", "Extremely High"
    
    def _get_health_advisory(self, aqi_value: float, category: str) -> str:
        """Get health advisory message based on AQI."""
        advisories = {
            "Good": "Air quality is satisfactory and poses little or no health risk.",
            "Moderate": "Air quality is acceptable for most people. Sensitive individuals may experience minor issues.",
            "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. General public not likely affected.",
            "Unhealthy": "Everyone may experience health effects. Sensitive groups may experience serious effects.",
            "Very Unhealthy": "Health alert - everyone may experience serious health effects.",
            "Hazardous": "Emergency conditions - everyone is likely to be affected."
        }
        return advisories.get(category, "Unknown health risk")
    
    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze pollution trends over time."""
        try:
            temporal_trends = {}
            
            # Hourly patterns
            if 'pm25_ugm3' in df.columns:
                hourly_pm25 = df.groupby(df['timestamp'].dt.hour)['pm25_ugm3'].mean()
                temporal_trends['hourly_pm25'] = hourly_pm25.to_dict()
                temporal_trends['peak_pollution_hour'] = hourly_pm25.idxmax()
            
            # Daily trends
            daily_aqi = df.groupby(df['timestamp'].dt.date)['aqi'].mean() if 'aqi' in df.columns else None
            if daily_aqi is not None:
                temporal_trends['daily_aqi_trend'] = {
                    'slope': self._calculate_trend_slope(daily_aqi),
                    'mean': daily_aqi.mean(),
                    'std': daily_aqi.std()
                }
            
            # Weekly patterns
            weekly_patterns = df.groupby(df['timestamp'].dt.day_name()).agg({
                'pm25_ugm3': 'mean' if 'pm25_ugm3' in df.columns else 'count',
                'no2_ugm3': 'mean' if 'no2_ugm3' in df.columns else 'count'
            })
            temporal_trends['weekly_patterns'] = weekly_patterns.to_dict('index')
            
            return temporal_trends
            
        except Exception:
            return {}
    
    def _analyze_weather_correlation(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze correlation between weather and pollution."""
        try:
            weather_correlation = {}
            
            # Pollution levels by weather condition
            weather_pollution = df.groupby('weather_condition').agg({
                'pm25_ugm3': 'mean' if 'pm25_ugm3' in df.columns else 'count',
                'no2_ugm3': 'mean' if 'no2_ugm3' in df.columns else 'count',
                'aqi': 'mean' if 'aqi' in df.columns else 'count'
            })
            weather_correlation['pollution_by_weather'] = weather_pollution.to_dict('index')
            
            # Weather impact factors
            if 'pm25_ugm3' in df.columns:
                clear_pm25 = df[df['weather_condition'] == 'Clear']['pm25_ugm3'].mean()
                rain_pm25 = df[df['weather_condition'] == 'Rain']['pm25_ugm3'].mean()
                
                if clear_pm25 > 0:
                    rain_reduction = ((clear_pm25 - rain_pm25) / clear_pm25) * 100
                    weather_correlation['rain_pollution_reduction'] = rain_reduction
            
            return weather_correlation
            
        except Exception:
            return {}
    
    def _analyze_traffic_correlation(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlation between traffic and pollution."""
        try:
            correlations = {}
            
            # Traffic-pollution correlations
            if 'pm25_ugm3' in df.columns:
                pm25_traffic_corr = df['pm25_ugm3'].corr(df['vehicle_count_estimated'])
                correlations['pm25_traffic_correlation'] = pm25_traffic_corr
            
            if 'no2_ugm3' in df.columns:
                no2_traffic_corr = df['no2_ugm3'].corr(df['vehicle_count_estimated'])
                correlations['no2_traffic_correlation'] = no2_traffic_corr
            
            # High traffic impact
            high_traffic_threshold = df['vehicle_count_estimated'].quantile(0.8)
            high_traffic_data = df[df['vehicle_count_estimated'] >= high_traffic_threshold]
            low_traffic_data = df[df['vehicle_count_estimated'] < high_traffic_threshold]
            
            if not high_traffic_data.empty and not low_traffic_data.empty:
                if 'pm25_ugm3' in df.columns:
                    high_traffic_pm25 = high_traffic_data['pm25_ugm3'].mean()
                    low_traffic_pm25 = low_traffic_data['pm25_ugm3'].mean()
                    correlations['high_traffic_pm25_increase'] = ((high_traffic_pm25 - low_traffic_pm25) / low_traffic_pm25) * 100
            
            return correlations
            
        except Exception:
            return {}
    
    def _analyze_spatial_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze spatial patterns across monitoring stations."""
        try:
            spatial_analysis = {}
            
            # Station comparison
            station_stats = df.groupby('station_id').agg({
                'pm25_ugm3': ['mean', 'std'] if 'pm25_ugm3' in df.columns else 'count',
                'no2_ugm3': ['mean', 'std'] if 'no2_ugm3' in df.columns else 'count',
                'aqi': ['mean', 'max'] if 'aqi' in df.columns else 'count'
            })
            
            # Flatten column names
            station_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                   for col in station_stats.columns]
            
            spatial_analysis['station_statistics'] = station_stats.to_dict('index')
            
            # Identify hotspots
            if 'aqi' in df.columns:
                station_avg_aqi = df.groupby('station_id')['aqi'].mean()
                hotspots = station_avg_aqi[station_avg_aqi > 100].index.tolist()
                spatial_analysis['pollution_hotspots'] = hotspots
            
            return spatial_analysis
            
        except Exception:
            return {}
    
    def _analyze_health_impact_trends(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze trends in health impact metrics."""
        try:
            health_trends = {}
            
            # Health risk distribution
            if 'health_risk' in df.columns:
                risk_distribution = df['health_risk'].value_counts(normalize=True) * 100
                health_trends['health_risk_distribution'] = risk_distribution.to_dict()
            
            # Unhealthy days count
            if 'aqi' in df.columns:
                daily_aqi = df.groupby(df['timestamp'].dt.date)['aqi'].mean()
                unhealthy_days = (daily_aqi > 100).sum()
                total_days = len(daily_aqi)
                health_trends['unhealthy_days_percent'] = (unhealthy_days / total_days) * 100
            
            # Health cost trends (if available)
            if 'health_cost_usd_daily' in df.columns:
                monthly_health_costs = df.groupby(df['timestamp'].dt.to_period('M'))['health_cost_usd_daily'].sum()
                health_trends['monthly_health_costs'] = monthly_health_costs.to_dict()
            
            return health_trends
            
        except Exception:
            return {}
    
    def _calculate_pm25_health_costs(self, pm25_data: pd.Series, population: int) -> Dict[str, float]:
        """Calculate health costs from PM2.5 exposure."""
        try:
            avg_pm25 = pm25_data.mean()
            
            # Health cost factors (simplified estimates, dollars per μg/m³ per person per year)
            mortality_cost_factor = 50  # Premature mortality
            morbidity_cost_factor = 20  # Hospital visits, medication, etc.
            
            # Calculate annual costs
            pm25_mortality_cost = avg_pm25 * mortality_cost_factor * population
            pm25_morbidity_cost = avg_pm25 * morbidity_cost_factor * population
            
            return {
                'pm25_mortality_cost': pm25_mortality_cost,
                'pm25_morbidity_cost': pm25_morbidity_cost
            }
            
        except Exception:
            return {}
    
    def _calculate_no2_health_costs(self, no2_data: pd.Series, population: int) -> Dict[str, float]:
        """Calculate health costs from NO2 exposure."""
        try:
            avg_no2 = no2_data.mean()
            
            # Health cost factors for NO2
            respiratory_cost_factor = 15  # Respiratory issues
            cardiovascular_cost_factor = 10  # Cardiovascular effects
            
            no2_respiratory_cost = avg_no2 * respiratory_cost_factor * population
            no2_cardiovascular_cost = avg_no2 * cardiovascular_cost_factor * population
            
            return {
                'no2_respiratory_cost': no2_respiratory_cost,
                'no2_cardiovascular_cost': no2_cardiovascular_cost
            }
            
        except Exception:
            return {}
    
    def _calculate_trend_slope(self, data: pd.Series) -> float:
        """Calculate trend slope using simple linear regression."""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            y = data.values
            
            # Simple linear regression
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
            
            return slope
            
        except Exception:
            return 0.0
    
    def _calculate_pollution_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate summary pollution statistics."""
        try:
            stats = {}
            
            # AQI statistics
            if 'aqi' in df.columns:
                stats['aqi'] = {
                    'mean': df['aqi'].mean(),
                    'median': df['aqi'].median(),
                    'max': df['aqi'].max(),
                    'std': df['aqi'].std(),
                    'days_over_100': (df.groupby(df['timestamp'].dt.date)['aqi'].mean() > 100).sum()
                }
            
            # Pollutant statistics
            pollutants = ['pm25_ugm3', 'no2_ugm3', 'o3_ugm3', 'co_mgm3']
            for pollutant in pollutants:
                if pollutant in df.columns:
                    stats[pollutant] = {
                        'mean': df[pollutant].mean(),
                        'median': df[pollutant].median(),
                        'max': df[pollutant].max(),
                        'std': df[pollutant].std()
                    }
            
            return stats
            
        except Exception:
            return {}
    
    def _generate_pollution_recommendations(self, df: pd.DataFrame, 
                                          aqi_metrics: AQIMetrics) -> List[str]:
        """Generate pollution management recommendations."""
        recommendations = []
        
        try:
            # AQI-based recommendations
            if aqi_metrics.overall_aqi > 150:
                recommendations.append("Critical air quality - implement emergency pollution reduction measures")
            elif aqi_metrics.overall_aqi > 100:
                recommendations.append("Unhealthy air quality - consider traffic restrictions and public health advisories")
            elif aqi_metrics.overall_aqi > 50:
                recommendations.append("Moderate air quality - monitor sensitive groups and consider preventive measures")
            
            # Dominant pollutant recommendations
            if aqi_metrics.dominant_pollutant == 'PM25':
                recommendations.append("PM2.5 is the dominant pollutant - focus on dust control and vehicle emission reduction")
            elif aqi_metrics.dominant_pollutant == 'NO2':
                recommendations.append("NO2 levels are high - prioritize diesel vehicle emission controls and traffic flow optimization")
            
            # Traffic-based recommendations
            if 'vehicle_count_estimated' in df.columns:
                high_traffic_hours = df.groupby(df['timestamp'].dt.hour)['vehicle_count_estimated'].mean()
                peak_hour = high_traffic_hours.idxmax()
                recommendations.append(f"Highest traffic at hour {peak_hour} - consider congestion pricing to reduce emissions")
            
            # Weather-based recommendations
            if 'weather_condition' in df.columns:
                fog_days = len(df[df['weather_condition'] == 'Fog'])
                if fog_days > len(df) * 0.1:  # More than 10% fog days
                    recommendations.append("Frequent fog conditions trap pollutants - enhance air quality monitoring during fog events")
            
            # Seasonal recommendations
            if 'timestamp' in df.columns:
                monthly_pollution = df.groupby(df['timestamp'].dt.month)['aqi'].mean() if 'aqi' in df.columns else None
                if monthly_pollution is not None:
                    worst_month = monthly_pollution.idxmax()
                    recommendations.append(f"Month {worst_month} shows highest pollution - plan seasonal emission reduction strategies")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Continue regular air quality monitoring and maintain current emission control measures")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
            recommendations.append("Manual analysis recommended - automated recommendations unavailable")
        
        return recommendations


class EmissionsCalculator:
    """
    Specialized calculator for vehicle emissions estimation.
    """
    
    def __init__(self):
        # EPA emission factors (grams per mile)
        self.emission_factors = {
            'passenger_car': {
                'co2': 411,      # grams CO2 per mile
                'nox': 0.2,      # grams NOx per mile
                'pm25': 0.01,    # grams PM2.5 per mile
                'co': 4.2,       # grams CO per mile
                'voc': 0.4       # grams VOC per mile
            },
            'light_truck': {
                'co2': 533,
                'nox': 0.3,
                'pm25': 0.015,
                'co': 5.8,
                'voc': 0.5
            },
            'heavy_truck': {
                'co2': 1690,
                'nox': 4.5,
                'pm25': 0.1,
                'co': 15.5,
                'voc': 1.4
            },
            'bus': {
                'co2': 1325,
                'nox': 11.0,
                'pm25': 0.3,
                'co': 8.9,
                'voc': 1.1
            },
            'motorcycle': {
                'co2': 180,
                'nox': 0.2,
                'pm25': 0.005,
                'co': 12.0,
                'voc': 1.0
            }
        }
    
    def calculate_fleet_emissions(self, vehicle_counts: Dict[str, int], 
                                distance_miles: float = 10) -> Dict[str, float]:
        """
        Calculate total emissions for a vehicle fleet.
        
        Args:
            vehicle_counts: Dictionary of vehicle type counts
            distance_miles: Average distance traveled
            
        Returns:
            Dictionary with total emissions by pollutant
        """
        try:
            total_emissions = {
                'co2_kg': 0, 'nox_kg': 0, 'pm25_kg': 0, 
                'co_kg': 0, 'voc_kg': 0
            }
            
            for vehicle_type, count in vehicle_counts.items():
                # Map vehicle type to emission factor category
                factor_key = self._map_vehicle_type(vehicle_type)
                factors = self.emission_factors.get(factor_key, self.emission_factors['passenger_car'])
                
                # Calculate emissions for this vehicle type
                for pollutant, factor in factors.items():
                    emission_kg = (count * distance_miles * factor) / 1000  # Convert g to kg
                    total_emissions[f'{pollutant}_kg'] += emission_kg
            
            return total_emissions
            
        except Exception as e:
            logger.error(f"Error calculating fleet emissions: {str(e)}")
            return {}
    
    def calculate_emission_reductions(self, baseline_emissions: Dict[str, float],
                                    reduction_scenarios: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate emission reductions for different scenarios.
        
        Args:
            baseline_emissions: Current emission levels
            reduction_scenarios: Dictionary of reduction percentages by scenario
            
        Returns:
            Dictionary with emission reductions by scenario
        """
        try:
            reductions = {}
            
            for scenario_name, reduction_pct in reduction_scenarios.items():
                scenario_reductions = {}
                
                for pollutant, baseline_value in baseline_emissions.items():
                    reduction_amount = baseline_value * (reduction_pct / 100)
                    new_value = baseline_value - reduction_amount
                    
                    scenario_reductions[pollutant] = {
                        'baseline': baseline_value,
                        'reduced': new_value,
                        'reduction_amount': reduction_amount,
                        'reduction_percent': reduction_pct
                    }
                
                reductions[scenario_name] = scenario_reductions
            
            return reductions
            
        except Exception as e:
            logger.error(f"Error calculating emission reductions: {str(e)}")
            return {}
    
    def _map_vehicle_type(self, vehicle_type: str) -> str:
        """Map vehicle type string to emission factor category."""
        vehicle_type_lower = vehicle_type.lower()
        
        if 'car' in vehicle_type_lower or 'sedan' in vehicle_type_lower:
            return 'passenger_car'
        elif 'truck' in vehicle_type_lower:
            if 'light' in vehicle_type_lower or 'pickup' in vehicle_type_lower:
                return 'light_truck'
            else:
                return 'heavy_truck'
        elif 'bus' in vehicle_type_lower:
            return 'bus'
        elif 'motorcycle' in vehicle_type_lower or 'bike' in vehicle_type_lower:
            return 'motorcycle'
        else:
            return 'passenger_car'  # Default


class AQICalculator:
    """
    Specialized calculator for Air Quality Index calculations.
    """
    
    def __init__(self):
        # AQI breakpoints based on EPA standards
        self.aqi_breakpoints = {
            'PM25': [
                (0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ],
            'NO2': [
                (0, 53, 0, 50),
                (54, 100, 51, 100),
                (101, 360, 101, 150),
                (361, 649, 151, 200),
                (650, 1249, 201, 300),
                (1250, 1649, 301, 400),
                (1650, 2049, 401, 500)
            ],
            'O3': [
                (0, 54, 0, 50),
                (55, 70, 51, 100),
                (71, 85, 101, 150),
                (86, 105, 151, 200),
                (106, 200, 201, 300)
            ],
            'CO': [
                (0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 40.4, 301, 400),
                (40.5, 50.4, 401, 500)
            ]
        }
    
    def calculate_pm25_aqi(self, concentration: float) -> float:
        """Calculate AQI for PM2.5 concentration."""
        return self._calculate_aqi(concentration, 'PM25')
    
    def calculate_no2_aqi(self, concentration: float) -> float:
        """Calculate AQI for NO2 concentration."""
        return self._calculate_aqi(concentration, 'NO2')
    
    def calculate_o3_aqi(self, concentration: float) -> float:
        """Calculate AQI for O3 concentration."""
        return self._calculate_aqi(concentration, 'O3')
    
    def calculate_co_aqi(self, concentration: float) -> float:
        """Calculate AQI for CO concentration."""
        return self._calculate_aqi(concentration, 'CO')
    
    def _calculate_aqi(self, concentration: float, pollutant: str) -> float:
        """
        Calculate AQI using the standard AQI formula.
        
        AQI = ((I_hi - I_lo) / (C_hi - C_lo)) * (C - C_lo) + I_lo
        
        Where:
        - C = pollutant concentration
        - C_lo = concentration breakpoint ≤ C
        - C_hi = concentration breakpoint ≥ C
        - I_lo = AQI value corresponding to C_lo
        - I_hi = AQI value corresponding to C_hi
        """
        try:
            if concentration < 0:
                return 0
            
            breakpoints = self.aqi_breakpoints.get(pollutant, [])
            if not breakpoints:
                return 0
            
            # Find the appropriate breakpoint
            for c_lo, c_hi, i_lo, i_hi in breakpoints:
                if c_lo <= concentration <= c_hi:
                    # Calculate AQI using the standard formula
                    if c_hi == c_lo:  # Avoid division by zero
                        return i_lo
                    
                    aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
                    return round(aqi)
            
            # If concentration exceeds all breakpoints, use the highest AQI
            return 500
            
        except Exception as e:
            logger.warning(f"Error calculating AQI for {pollutant}: {str(e)}")
            return 0
    
    def get_aqi_description(self, aqi_value: float) -> Dict[str, str]:
        """Get detailed description for AQI value."""
        if aqi_value <= 50:
            return {
                'level': 'Good',
                'color': 'Green',
                'description': 'Air quality is satisfactory',
                'health_message': 'Air quality is considered satisfactory, and air pollution poses little or no risk.'
            }
        elif aqi_value <= 100:
            return {
                'level': 'Moderate',
                'color': 'Yellow',
                'description': 'Air quality is acceptable',
                'health_message': 'Air quality is acceptable for most people. However, sensitive people may experience minor issues.'
            }
        elif aqi_value <= 150:
            return {
                'level': 'Unhealthy for Sensitive Groups',
                'color': 'Orange',
                'description': 'Sensitive groups may experience health effects',
                'health_message': 'Members of sensitive groups may experience health effects. The general public is not likely to be affected.'
            }
        elif aqi_value <= 200:
            return {
                'level': 'Unhealthy',
                'color': 'Red',
                'description': 'Everyone may experience health effects',
                'health_message': 'Everyone may begin to experience health effects; sensitive groups may experience serious effects.'
            }
        elif aqi_value <= 300:
            return {
                'level': 'Very Unhealthy',
                'color': 'Purple',
                'description': 'Health warnings of emergency conditions',
                'health_message': 'Health alert: everyone may experience more serious health effects.'
            }
        else:
            return {
                'level': 'Hazardous',
                'color': 'Maroon',
                'description': 'Emergency conditions',
                'health_message': 'Health warnings of emergency conditions. The entire population is more likely to be affected.'
            }
    
    def calculate_composite_aqi(self, pollutant_concentrations: Dict[str, float]) -> Dict[str, any]:
        """
        Calculate composite AQI from multiple pollutants.
        
        Args:
            pollutant_concentrations: Dictionary of pollutant concentrations
            
        Returns:
            Dictionary with composite AQI analysis
        """
        try:
            individual_aqis = {}
            
            # Calculate AQI for each pollutant
            for pollutant, concentration in pollutant_concentrations.items():
                if pollutant.upper() in self.aqi_breakpoints:
                    aqi_value = self._calculate_aqi(concentration, pollutant.upper())
                    individual_aqis[pollutant] = aqi_value
            
            if not individual_aqis:
                return {}
            
            # Overall AQI is the maximum of individual pollutant AQIs
            overall_aqi = max(individual_aqis.values())
            dominant_pollutant = max(individual_aqis, key=individual_aqis.get)
            
            # Get description
            aqi_info = self.get_aqi_description(overall_aqi)
            
            return {
                'overall_aqi': overall_aqi,
                'dominant_pollutant': dominant_pollutant,
                'individual_aqis': individual_aqis,
                'aqi_level': aqi_info['level'],
                'health_message': aqi_info['health_message'],
                'color_code': aqi_info['color']
            }
            
        except Exception as e:
            logger.error(f"Error calculating composite AQI: {str(e)}")
            return {}"""
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
