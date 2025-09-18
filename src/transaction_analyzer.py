def calculate_revenue_metrics(self, df: pd.DataFrame,
                                period: str = 'daily') -> RevenueMetrics:
        """
        Calculate comprehensive revenue metrics.
        
        Args:
            df: DataFrame with transaction data
            period: Analysis period ('daily', 'weekly', 'monthly')
            
        Returns:
            RevenueMetrics object with analysis results
        """
        try:
            # Validate input data
            if not self.validator.validate_transaction_data(df):
                raise ValueError("Invalid transaction data format")
            
            # Filter successful transactions
            successful_tx = df[df.get('transaction_success', True)]
            
            if successful_tx.empty:
                logger.warning("No successful transactions found")
                return RevenueMetrics(0, 0, 0, 0, "N/A", {}, 0)
            
            # Basic metrics
            total_revenue = successful_tx['final_toll'].sum()
            transaction_count = len(successful_tx)
            avg_transaction_value = total_revenue / transaction_count if transaction_count > 0 else 0
            
            # Success rate
            success_rate = len(successful_tx) / len(df) * 100 if len(df) > 0 else 0
            
            # Peak revenue hour
            if 'timestamp' in successful_tx.columns:
                hourly_revenue = successful_tx.groupby(successful_tx['timestamp'].dt.hour)['final_toll'].sum()
                peak_revenue_hour = f"{hourly_revenue.idxmax()}:00"
            else:
                peak_revenue_hour = "N/A"
            
            # Payment method breakdown
            payment_breakdown = {}
            if 'payment_method' in successful_tx.columns:
                payment_breakdown = successful_tx.groupby('payment_method')['final_toll'].sum().to_dict()
            
            # Revenue growth rate (comparing with previous period)
            revenue_growth_rate = self._calculate_growth_rate(successful_tx, period)
            
            metrics = RevenueMetrics(
                total_revenue=total_revenue,
                transaction_count=transaction_count,
                avg_transaction_value=avg_transaction_value,
                revenue_growth_rate=revenue_growth_rate,
                peak_revenue_hour=peak_revenue_hour,
                payment_method_breakdown=payment_breakdown,
                success_rate=success_rate
            )
            
            logger.info(f"Revenue metrics calculated: ${total_revenue:.2f} from {transaction_count} transactions")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating revenue metrics: {str(e)}")
            raise
    
    def analyze_payment_methods(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze payment method performance and patterns.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with payment method analysis
        """
        try:
            if 'payment_method' not in df.columns:
                logger.warning("No payment_method column found")
                return {}
            
            # Payment method distribution
            method_counts = df['payment_method'].value_counts()
            method_percentages = df['payment_method'].value_counts(normalize=True) * 100
            
            # Success rates by payment method
            success_rates = {}
            if 'transaction_success' in df.columns:
                success_rates = df.groupby('payment_method')['transaction_success'].mean() * 100
            
            # Average processing time by method
            processing_times = {}
            if 'processing_time_sec' in df.columns:
                processing_times = df.groupby('payment_method')['processing_time_sec'].mean()
            
            # Revenue by payment method
            revenue_by_method = {}
            if 'final_toll' in df.columns:
                revenue_by_method = df[df.get('transaction_success', True)].groupby('payment_method')['final_toll'].sum()
            
            # Transaction value distribution by method
            value_stats = {}
            if 'final_toll' in df.columns:
                value_stats = df.groupby('payment_method')['final_toll'].agg(['mean', 'std', 'min', 'max']).to_dict('index')
            
            results = {
                'method_distribution': method_counts.to_dict(),
                'method_percentages': method_percentages.to_dict(),
                'success_rates': success_rates.to_dict() if success_rates else {},
                'avg_processing_times': processing_times.to_dict() if processing_times else {},
                'revenue_by_method': revenue_by_method.to_dict() if not revenue_by_method.empty else {},
                'value_statistics': value_stats,
                'most_popular_method': method_counts.index[0] if not method_counts.empty else None,
                'fastest_method': min(processing_times.items(), key=lambda x: x[1])[0] if processing_times else None
            }
            
            logger.info(f"Analyzed {len(method_counts)} payment methods")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing payment methods: {str(e)}")
            raise
    
    def detect_fraud(self, df: pd.DataFrame) -> List[FraudAlert]:
        """
        Detect potentially fraudulent transactions.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            List of FraudAlert objects
        """
        try:
            alerts = []
            
            if df.empty:
                return alerts
            
            # High-value transaction detection
            if 'final_toll' in df.columns:
                high_value_threshold = self.config['fraud_thresholds']['max_transaction_amount']
                high_value_tx = df[df['final_toll'] > high_value_threshold]
                
                for _, tx in high_value_tx.iterrows():
                    alert = FraudAlert(
                        transaction_id=tx.get('transaction_id', 'Unknown'),
                        alert_type='High Value Transaction',
                        risk_score=min(100, (tx['final_toll'] / high_value_threshold) * 50),
                        details=f"Transaction amount ${tx['final_toll']:.2f} exceeds threshold",
                        timestamp=tx.get('timestamp', datetime.now())
                    )
                    alerts.append(alert)
            
            # Velocity-based detection (too many transactions in short time)
            if 'timestamp' in df.columns and 'vehicle_id' in df.columns:
                alerts.extend(self._detect_velocity_fraud(df))
            
            # Duplicate transaction detection
            if 'vehicle_id' in df.columns and 'final_toll' in df.columns:
                alerts.extend(self._detect_duplicate_transactions(df))
            
            # Failed transaction patterns
            if 'transaction_success' in df.columns:
                alerts.extend(self._detect_suspicious_failures(df))
            
            logger.info(f"Detected {len(alerts)} potential fraud alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Error detecting fraud: {str(e)}")
            return []
    
    def analyze_pricing_impact(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze the impact of pricing strategies on revenue and traffic.
        
        Args:
            df: DataFrame with transaction data including pricing information
            
        Returns:
            Dictionary with pricing impact analysis
        """
        try:
            results = {}
            
            # Peak pricing analysis
            if 'peak_multiplier' in df.columns:
                peak_analysis = df.groupby('peak_multiplier').agg({
                    'final_toll': ['sum', 'count', 'mean'],
                    'transaction_success': 'mean' if 'transaction_success' in df.columns else 'count'
                })
                
                results['peak_pricing_impact'] = peak_analysis.to_dict()
            
            # Discount analysis
            if 'discount_type' in df.columns:
                discount_analysis = df.groupby('discount_type').agg({
                    'final_toll': ['sum', 'count', 'mean'],
                    'discount_factor': 'mean' if 'discount_factor' in df.columns else 'count'
                })
                
                # Calculate discount impact on revenue
                no_discount_revenue = df[df['discount_type'] == 'None']['final_toll'].sum() if 'None' in df['discount_type'].values else 0
                total_discount_revenue = df[df['discount_type'] != 'None']['final_toll'].sum()
                
                # Estimate what revenue would have been without discounts
                discounted_tx = df[df['discount_type'] != 'None']
                if not discounted_tx.empty and 'discount_factor' in df.columns:
                    potential_revenue = (discounted_tx['final_toll'] / discounted_tx['discount_factor']).sum()
                    discount_cost = potential_revenue - total_discount_revenue
                else:
                    discount_cost = 0
                
                results['discount_analysis'] = {
                    'breakdown': discount_analysis.to_dict(),
                    'total_discount_cost': discount_cost,
                    'discount_transactions': len(discounted_tx)
                }
            
            # Price elasticity estimation
            if 'vehicle_count' in df.columns and 'final_toll' in df.columns:
                results['price_elasticity'] = self._estimate_price_elasticity(df)
            
            # Optimal pricing recommendations
            results['pricing_recommendations'] = self._generate_pricing_recommendations(df)
            
            logger.info("Pricing impact analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing pricing impact: {str(e)}")
            return {}
    
    def calculate_customer_lifetime_value(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate customer lifetime value based on transaction history.
        
        Args:
            df: DataFrame with customer transaction data
            
        Returns:
            Dictionary with CLV metrics
        """
        try:
            if 'customer_id' not in df.columns:
                logger.warning("No customer_id column found for CLV calculation")
                return {}
            
            # Customer metrics
            customer_metrics = df.groupby('customer_id').agg({
                'final_toll': ['sum', 'mean', 'count'],
                'timestamp': ['min', 'max']
            })
            
            # Flatten columns
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Calculate CLV components
            customer_metrics['total_revenue'] = customer_metrics['final_toll_sum']
            customer_metrics['avg_transaction'] = customer_metrics['final_toll_mean']
            customer_metrics['transaction_count'] = customer_metrics['final_toll_count']
            customer_metrics['customer_lifespan_days'] = (
                customer_metrics['timestamp_max'] - customer_metrics['timestamp_min']
            ).dt.days
            
            # Simple CLV calculation
            customer_metrics['clv'] = (
                customer_metrics['avg_transaction'] * 
                customer_metrics['transaction_count'] * 
                (customer_metrics['customer_lifespan_days'] / 365 + 1)  # Assume at least 1 year
            )
            
            # Aggregate statistics
            clv_stats = {
                'avg_clv': customer_metrics['clv'].mean(),
                'median_clv': customer_metrics['clv'].median(),
                'total_customers': len(customer_metrics),
                'high_value_customers': len(customer_metrics[customer_metrics['clv'] > customer_metrics['clv'].quantile(0.8)]),
                'clv_std': customer_metrics['clv'].std()
            }
            
            logger.info(f"Calculated CLV for {len(customer_metrics)} customers")
            return clv_stats
            
        except Exception as e:
            logger.error(f"Error calculating CLV: {str(e)}")
            return {}
    
    def generate_revenue_report(self, df: pd.DataFrame, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, any]:
        """
        Generate comprehensive revenue report.
        
        Args:
            df: DataFrame with transaction data
            start_date: Start date for report (YYYY-MM-DD)
            end_date: End date for report (YYYY-MM-DD)
            
        Returns:
            Dictionary with complete revenue analysis
        """
        try:
            # Filter by date range if provided
            if start_date or end_date:
                df = self.processor.filter_by_date_range(df, start_date, end_date)
            
            if df.empty:
                logger.warning("No data available for revenue report")
                return {}
            
            # Core metrics
            revenue_metrics = self.calculate_revenue_metrics(df)
            payment_analysis = self.analyze_payment_methods(df)
            pricing_analysis = self.analyze_pricing_impact(df)
            
            # Time-based analysis
            time_analysis = {}
            if 'timestamp' in df.columns:
                time_analysis = self._analyze_revenue_by_time(df)
            
            # Vehicle type analysis
            vehicle_analysis = {}
            if 'vehicle_type' in df.columns:
                vehicle_analysis = df.groupby('vehicle_type')['final_toll'].agg(['sum', 'count', 'mean']).to_dict('index')
            
            # Lane performance
            lane_analysis = {}
            if 'lane_id' in df.columns:
                lane_analysis = df.groupby('lane_id')['final_toll'].agg(['sum', 'count', 'mean']).to_dict('index')
            
            report = {
                'report_period': {
                    'start_date': start_date or df['timestamp'].min().strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'N/A',
                    'end_date': end_date or df['timestamp'].max().strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'N/A'
                },
                'revenue_metrics': revenue_metrics.__dict__,
                'payment_analysis': payment_analysis,
                'pricing_analysis': pricing_analysis,
                'time_analysis': time_analysis,
                'vehicle_analysis': vehicle_analysis,
                'lane_analysis': lane_analysis,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info("Revenue report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating revenue report: {str(e)}")
            raise
    
    def _calculate_growth_rate(self, df: pd.DataFrame, period: str) -> float:
        """Calculate revenue growth rate for the specified period."""
        try:
            if 'timestamp' not in df.columns or df.empty:
                return 0.0
            
            # Group by period
            if period == 'daily':
                grouped = df.groupby(df['timestamp'].dt.date)['final_toll'].sum()
            elif period == 'weekly':
                grouped = df.groupby(df['timestamp'].dt.isocalendar().week)['final_toll'].sum()
            elif period == 'monthly':
                grouped = df.groupby(df['timestamp'].dt.to_period('M'))['final_toll'].sum()
            else:
                return 0.0
            
            if len(grouped) < 2:
                return 0.0
            
            # Calculate growth rate
            current_period = grouped.iloc[-1]
            previous_period = grouped.iloc[-2]
            
            if previous_period == 0:
                return 0.0
            
            growth_rate = ((current_period - previous_period) / previous_period) * 100
            return growth_rate
            
        except Exception:
            return 0.0
    
    def _detect_velocity_fraud(self, df: pd.DataFrame) -> List[FraudAlert]:
        """Detect fraud based on transaction velocity."""
        alerts = []
        
        try:
            velocity_threshold = self.config['fraud_thresholds']['velocity_threshold']
            
            # Group by vehicle_id and check transaction frequency
            for vehicle_id, group in df.groupby('vehicle_id'):
                if len(group) < 2:
                    continue
                
                # Sort by timestamp
                group = group.sort_values('timestamp')
                
                # Calculate time differences
                time_diffs = group['timestamp'].diff().dt.total_seconds() / 60  # Convert to minutes
                
                # Check for high velocity
                rapid_transactions = time_diffs < (60 / velocity_threshold)  # Less than threshold interval
                
                if rapid_transactions.sum() > 0:
                    alert = FraudAlert(
                        transaction_id=f"Vehicle_{vehicle_id}",
                        alert_type='High Velocity',
                        risk_score=min(100, rapid_transactions.sum() * 20),
                        details=f"Vehicle {vehicle_id} has {rapid_transactions.sum()} rapid transactions",
                        timestamp=group['timestamp'].iloc[-1]
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error in velocity fraud detection: {str(e)}")
        
        return alerts
    
    def _detect_duplicate_transactions(self, df: pd.DataFrame) -> List[FraudAlert]:
        """Detect potential duplicate transactions."""
        alerts = []
        
        try:
            tolerance_minutes = self.config['fraud_thresholds']['duplicate_tolerance_minutes']
            
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            
            # Check for duplicates within tolerance window
            for i in range(len(df_sorted) - 1):
                current = df_sorted.iloc[i]
                next_tx = df_sorted.iloc[i + 1]
                
                # Check if same vehicle and similar amount within time window
                time_diff = (next_tx['timestamp'] - current['timestamp']).total_seconds() / 60
                amount_diff = abs(current['final_toll'] - next_tx['final_toll'])
                
                if (current.get('vehicle_id') == next_tx.get('vehicle_id') and
                    time_diff <= tolerance_minutes and
                    amount_diff < 0.01):  # Same amount
                    
                    alert = FraudAlert(
                        transaction_id=current.get('transaction_id', 'Unknown'),
                        alert_type='Potential Duplicate',
                        risk_score=70,
                        details=f"Similar transaction within {time_diff:.1f} minutes",
                        timestamp=current.get('timestamp', datetime.now())
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error in duplicate detection: {str(e)}")
        
        return alerts
    
    def _detect_suspicious_failures(self, df: pd.DataFrame) -> List[FraudAlert]:
        """Detect suspicious patterns in failed transactions."""
        alerts = []
        
        try:
            # Group failed transactions by vehicle_id
            failed_tx = df[df['transaction_success'] == False]
            
            if failed_tx.empty:
                return alerts
            
            failure_counts = failed_tx.groupby('vehicle_id').size()
            suspicious_vehicles = failure_counts[failure_counts >= 5]  # 5+ failures
            
            for vehicle_id, failure_count in suspicious_vehicles.items():
                alert = FraudAlert(
                    transaction_id=f"Vehicle_{vehicle_id}",
                    alert_type='Suspicious Failures',
                    risk_score=min(100, failure_count * 10),
                    details=f"Vehicle {vehicle_id} has {failure_count} failed transactions",
                    timestamp=failed_tx[failed_tx['vehicle_id'] == vehicle_id]['timestamp'].max()
                )
                alerts.append(alert)
        
        except Exception as e:
            logger.warning(f"Error in failure pattern detection: {str(e)}")
        
        return alerts
    
    def _estimate_price_elasticity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimate price elasticity of demand."""
        try:
            # Group by price ranges and calculate demand
            df['price_range'] = pd.cut(df['final_toll'], bins=10, labels=False)
            
            elasticity_data = df.groupby('price_range').agg({
                'final_toll': 'mean',
                'vehicle_count': 'sum'
            })
            
            if len(elasticity_data) < 2:
                return {'elasticity': 0.0}
            
            # Simple elasticity calculation
            price_change = elasticity_data['final_toll'].pct_change()
            quantity_change = elasticity_data['vehicle_count'].pct_change()
            
            # Avoid division by zero
            valid_elasticity = (quantity_change / price_change).replace([np.inf, -np.inf], np.nan).dropna()
            
            if valid_elasticity.empty:
                return {'elasticity': 0.0}
            
            avg_elasticity = valid_elasticity.mean()
            
            return {
                'elasticity': avg_elasticity,
                'interpretation': 'elastic' if abs(avg_elasticity) > 1 else 'inelastic'
            }
            
        except Exception:
            return {'elasticity': 0.0}
    
    def _generate_pricing_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate pricing strategy recommendations."""
        recommendations = []
        
        try:
            # Analyze peak vs off-peak performance
            if 'peak_multiplier' in df.columns:
                peak_revenue = df[df['peak_multiplier'] > 1]['final_toll'].sum()
                regular_revenue = df[df['peak_multiplier'] == 1]['final_toll'].sum()
                
                if peak_revenue > regular_revenue * 0.5:  # Peak revenue is significant
                    recommendations.append("Peak pricing is effective - consider expanding peak hours")
                else:
                    recommendations.append("Peak pricing may be too aggressive - consider reducing multiplier")
            
            # Analyze payment method preferences
            if 'payment_method' in df.columns:
                method_revenue = df.groupby('payment_method')['final_toll'].sum()
                top_method = method_revenue.idxmax()
                recommendations.append(f"Focus on optimizing {top_method} payment processing")
            
            # Vehicle type analysis
            if 'vehicle_type' in df.columns:
                vehicle_revenue = df.groupby('vehicle_type')['final_toll'].mean()
                commercial_vehicles = ['Truck', 'Bus', 'Trailer']
                commercial_avg = vehicle_revenue[vehicle_revenue.index.isin(commercial_vehicles)].mean()
                personal_avg = vehicle_revenue[~vehicle_revenue.index.isin(commercial_vehicles)].mean()
                
                if commercial_avg > personal_avg * 2:
                    recommendations.append("Consider tiered commercial vehicle pricing")
            
        except Exception as e:
            logger.warning(f"Error generating pricing recommendations: {str(e)}")
            recommendations.append("Unable to generate specific recommendations - more data needed")
        
        return recommendations
    
    def _analyze_revenue_by_time(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze revenue patterns by time."""
        try:
            time_analysis = {}
            
            # Hourly analysis
            hourly_revenue = df.groupby(df['timestamp'].dt.hour)['final_toll'].sum()
            time_analysis['hourly_revenue'] = hourly_revenue.to_dict()
            time_analysis['peak_hour'] = hourly_revenue.idxmax()
            time_analysis['low_hour'] = hourly_revenue.idxmin()
            
            # Daily analysis
            daily_revenue = df.groupby(df['timestamp'].dt.date)['final_toll'].sum()
            time_analysis['daily_revenue_stats'] = {
                'mean': daily_revenue.mean(),
                'std': daily_revenue.std(),
                'min': daily_revenue.min(),
                'max': daily_revenue.max()
            }
            
            # Weekly pattern
            weekly_revenue = df.groupby(df['timestamp'].dt.day_name())['final_toll'].sum()
            time_analysis['weekly_pattern'] = weekly_revenue.to_dict()
            
            return time_analysis
            
        except Exception as e:
            logger.warning(f"Error in time-based analysis: {str(e)}")
            return {}


class RevenueOptimizer:
    """
    Specialized class for revenue optimization and pricing strategies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.datetime_utils = DateTimeUtils()
    
    def optimize_pricing_strategy(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Recommend optimal pricing strategies based on data analysis.
        
        Args:
            df: Transaction data DataFrame
            
        Returns:
            Dictionary with optimization recommendations
        """
        try:
            optimization_results = {
                'current_performance': self._analyze_current_performance(df),
                'pricing_recommendations': self._generate_pricing_strategies(df),
                'revenue_forecast': self._forecast_revenue_impact(df),
                'implementation_plan': self._create_implementation_plan(df)
            }
            
            logger.info("Pricing optimization analysis completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in pricing optimization: {str(e)}")
            return {}
    
    def _analyze_current_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze current pricing performance."""
        try:
            performance = {
                'total_revenue': df['final_toll'].sum(),
                'avg_transaction': df['final_toll'].mean(),
                'transaction_volume': len(df),
                'success_rate': df.get('transaction_success', pd.Series([True] * len(df))).mean() * 100
            }
            
            if 'peak_multiplier' in df.columns:
                performance['peak_premium'] = df[df['peak_multiplier'] > 1]['final_toll'].mean()
                performance['regular_rate'] = df[df['peak_multiplier'] == 1]['final_toll'].mean()
            
            return performance
            
        except Exception:
            return {}
    
    def _generate_pricing_strategies(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """Generate specific pricing strategy recommendations."""
        strategies = []
        
        try:
            # Dynamic pricing strategy
            if 'timestamp' in df.columns:
                strategies.append({
                    'name': 'Dynamic Congestion Pricing',
                    'description': 'Adjust prices based on real-time traffic conditions',
                    'expected_impact': '15-25% revenue increase',
                    'implementation_complexity': 'High'
                })
            
            # Loyalty pricing
            strategies.append({
                'name': 'Frequent User Discounts',
                'description': 'Provide discounts for regular commuters',
                'expected_impact': '10-15% volume increase',
                'implementation_complexity': 'Medium'
            })
            
            # Vehicle type optimization
            if 'vehicle_type' in df.columns:
                strategies.append({
                    'name': 'Vehicle Type Optimization',
                    'description': 'Optimize pricing tiers for different vehicle types',
                    'expected_impact': '5-10% revenue increase',
                    'implementation_complexity': 'Low'
                })
            
        except Exception as e:
            logger.warning(f"Error generating strategies: {str(e)}")
        
        return strategies
    
    def _forecast_revenue_impact(self, df: pd.DataFrame) -> Dict[str, float]:
        """Forecast revenue impact of optimization strategies."""
        try:
            current_revenue = df['final_toll'].sum()
            
            # Conservative estimates
            forecasts = {
                'baseline_revenue': current_revenue,
                'optimistic_scenario': current_revenue * 1.25,  # 25% increase
                'realistic_scenario': current_revenue * 1.15,   # 15% increase
                'conservative_scenario': current_revenue * 1.08  # 8% increase
            }
            
            return forecasts
            
        except Exception:
            return {}
    
    def _create_implementation_plan(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Create implementation plan for pricing optimization."""
        plan = [
            {
                'phase': 'Phase 1: Analysis',
                'duration': '2-4 weeks',
                'activities': 'Detailed traffic pattern analysis, price elasticity study',
                'deliverables': 'Comprehensive pricing analysis report'
            },
            {
                'phase': 'Phase 2: Pilot Implementation',
                'duration': '4-6 weeks',
                'activities': 'Implement dynamic pricing on select lanes',
                'deliverables': 'Pilot results and performance metrics'
            },
            {
                'phase': 'Phase 3: Full Rollout',
                'duration': '8-12 weeks',
                'activities': 'System-wide implementation of optimized pricing',
                'deliverables': 'Full deployment and monitoring dashboard'
            }
        ]
        
        return plan


class PaymentProcessor:
    """
    Specialized class for payment processing analysis and optimization.
    """
    
    def __init__(self):
        self.validator = ValidationUtils()
    
    def analyze_payment_performance(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze payment processing performance and identify issues.
        
        Args:
            df: Payment transaction DataFrame
            
        Returns:
            Dictionary with payment performance analysis
        """
        try:
            performance_analysis = {
                'processing_times': self._analyze_processing_times(df),
                'success_rates': self._analyze_success_rates(df),
                'error_patterns': self._analyze_error_patterns(df),
                'recommendations': self._generate_payment_recommendations(df)
            }
            
            logger.info("Payment performance analysis completed")
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing payment performance: {str(e)}")
            return {}
    
    def _analyze_processing_times(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze payment processing times by method."""
        try:
            if 'processing_time_sec' not in df.columns or 'payment_method' not in df.columns:
                return {}
            
            processing_stats = df.groupby('payment_method')['processing_time_sec'].agg([
                'mean', 'median', 'std', 'min', 'max'
            ]).to_dict('index')
            
            return processing_stats
            
        except Exception:
            return {}
    
    def _analyze_success_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze success rates by payment method."""
        try:
            if 'transaction_success' not in df.columns or 'payment_method' not in df.columns:
                return {}
            
            success_rates = df.groupby('payment_method')['transaction_success'].mean() * 100
            return success_rates.to_dict()
            
        except Exception:
            return {}
    
    def _analyze_error_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze error patterns in failed transactions."""
        try:
            failed_tx = df[df.get('transaction_success', True) == False]
            
            if failed_tx.empty:
                return {'total_errors': 0}
            
            error_analysis"""
Transaction Analyzer Module
==========================

Analyzes toll transaction data, revenue patterns, and payment processing
for comprehensive financial insights and optimization.

Classes:
--------
- TransactionAnalyzer: Main class for transaction analysis
- RevenueOptimizer: Revenue optimization and pricing strategies
- PaymentProcessor: Payment method analysis and fraud detection

Example Usage:
--------------
>>> analyzer = TransactionAnalyzer()
>>> metrics = analyzer.calculate_revenue_metrics(transaction_data)
>>> fraud_alerts = analyzer.detect_fraud(transaction_data)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import warnings

from .utils import DataProcessor, ValidationUtils, DateTimeUtils

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RevenueMetrics:
    """Data class for revenue analysis metrics."""
    total_revenue: float
    transaction_count: int
    avg_transaction_value: float
    revenue_growth_rate: float
    peak_revenue_hour: str
    payment_method_breakdown: Dict[str, float]
    success_rate: float

@dataclass
class FraudAlert:
    """Data class for fraud detection alerts."""
    transaction_id: str
    alert_type: str
    risk_score: float
    details: str
    timestamp: datetime

class TransactionAnalyzer:
    """
    Main class for analyzing toll transaction data and revenue patterns.
    
    Attributes:
        config (dict): Configuration parameters
        processor (DataProcessor): Data processing utilities
        validator (ValidationUtils): Data validation utilities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize TransactionAnalyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or self._get_default_config()
        self.processor = DataProcessor()
        self.validator = ValidationUtils()
        self.datetime_utils = DateTimeUtils()
        
        logger.info("TransactionAnalyzer initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'fraud_thresholds': {
                'max_transaction_amount': 1000.0,
                'velocity_threshold': 10,  # transactions per minute
                'duplicate_tolerance_minutes': 5
            },
            'revenue_targets': {
                'daily_target': 10000.0,
                'monthly_growth_rate': 0.05
            },
            'payment_processing': {
                'timeout_seconds': 30,
                'retry_attempts': 3,
                'success_threshold': 0.95
            }
        }
    
    def calculate_revenue_metrics(self, df: pd.DataFrame,
                                perio
