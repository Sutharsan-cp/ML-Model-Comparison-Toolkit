import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.linalg import inv
import warnings

class TimeSeriesMetrics:
    """Comprehensive time series forecasting metrics"""
    
    @staticmethod
    def _validate_inputs(y_true, y_pred, y_train=None):
        """Validate and convert inputs to numpy arrays"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match")
        
        if y_train is not None:
            y_train = np.asarray(y_train)
            
        return y_true, y_pred, y_train
    
    @staticmethod
    def _handle_zeros(y_true, y_pred, replace_value=1e-10):
        """Handle zeros for percentage-based metrics"""
        y_true_safe = np.where(y_true == 0, replace_value, y_true)
        y_pred_safe = np.where(y_pred == 0, replace_value, y_pred)
        return y_true_safe, y_pred_safe
    
    # =========================================================================
    # SCALED ERRORS
    # =========================================================================
    
    @staticmethod
    def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonal_period=1):
        """
        Mean Absolute Scaled Error (MASE)
        Scales error by in-sample naive forecast error
        """
        y_true, y_pred, y_train = TimeSeriesMetrics._validate_inputs(y_true, y_pred, y_train)
        
        if len(y_train) < 2:
            raise ValueError("Training data must have at least 2 observations for MASE")
        
        # Calculate MAE of forecasts
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        
        # Calculate MAE of seasonal naive forecast on training data
        if seasonal_period > 1 and len(y_train) > seasonal_period:
            # Seasonal naive forecast
            naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
        else:
            # Simple naive forecast (previous value)
            naive_errors = np.abs(np.diff(y_train))
        
        if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
            return float('inf')
        
        mae_naive = np.mean(naive_errors)
        
        return mae_forecast / mae_naive
    
    @staticmethod
    def root_mean_squared_scaled_error(y_true, y_pred, y_train, seasonal_period=1):
        """
        Root Mean Squared Scaled Error (RMSSE)
        Scales RMSE by in-sample naive forecast RMSE
        """
        y_true, y_pred, y_train = TimeSeriesMetrics._validate_inputs(y_true, y_pred, y_train)
        
        if len(y_train) < 2:
            raise ValueError("Training data must have at least 2 observations for RMSSE")
        
        # Calculate RMSE of forecasts
        rmse_forecast = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Calculate RMSE of seasonal naive forecast on training data
        if seasonal_period > 1 and len(y_train) > seasonal_period:
            # Seasonal naive forecast
            naive_errors_squared = (y_train[seasonal_period:] - y_train[:-seasonal_period]) ** 2
        else:
            # Simple naive forecast
            naive_errors_squared = np.diff(y_train) ** 2
        
        if len(naive_errors_squared) == 0 or np.mean(naive_errors_squared) == 0:
            return float('inf')
        
        rmse_naive = np.sqrt(np.mean(naive_errors_squared))
        
        return rmse_forecast / rmse_naive
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error (sMAPE)
        Symmetric version of MAPE that handles zero values better
        """
        y_true, y_pred, _ = TimeSeriesMetrics._validate_inputs(y_true, y_pred)
        y_true_safe, y_pred_safe = TimeSeriesMetrics._handle_zeros(y_true, y_pred)
        
        numerator = np.abs(y_true_safe - y_pred_safe)
        denominator = (np.abs(y_true_safe) + np.abs(y_pred_safe)) / 2
        
        return 100 * np.mean(numerator / denominator)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        """Standard MAPE for time series"""
        y_true, y_pred, _ = TimeSeriesMetrics._validate_inputs(y_true, y_pred)
        y_true_safe, _ = TimeSeriesMetrics._handle_zeros(y_true, y_pred)
        
        return 100 * np.mean(np.abs((y_true_safe - y_pred) / y_true_safe))
    
    # =========================================================================
    # PREDICTION INTERVAL METRICS
    # =========================================================================
    
    @staticmethod
    def prediction_interval_coverage(y_true, y_pred_lower, y_pred_upper):
        """
        Prediction Interval Coverage (PIC)
        Percentage of true values within prediction intervals
        """
        y_true = np.asarray(y_true)
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)
        
        if y_true.shape != y_pred_lower.shape or y_true.shape != y_pred_upper.shape:
            raise ValueError("All inputs must have the same shape")
        
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        return coverage
    
    @staticmethod
    def prediction_interval_width(y_pred_lower, y_pred_upper):
        """
        Average Prediction Interval Width
        Measures the sharpness of prediction intervals
        """
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)
        
        if y_pred_lower.shape != y_pred_upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape")
        
        return np.mean(y_pred_upper - y_pred_lower)
    
    @staticmethod
    def interval_score(y_true, y_pred_lower, y_pred_upper, alpha=0.05):
        """
        Interval Score - Proper scoring rule for prediction intervals
        Penalizes both wide intervals and missed coverage
        """
        y_true = np.asarray(y_true)
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)
        
        if y_true.shape != y_pred_lower.shape or y_true.shape != y_pred_upper.shape:
            raise ValueError("All inputs must have the same shape")
        
        width = y_pred_upper - y_pred_lower
        
        # Penalty for observations outside the interval
        penalty = (2 / alpha) * np.where(
            y_true < y_pred_lower, y_pred_lower - y_true,
            np.where(y_true > y_pred_upper, y_true - y_pred_upper, 0)
        )
        
        return np.mean(width + penalty)
    
    @staticmethod
    def coverage_width_based_criterion(y_true, y_pred_lower, y_pred_upper, alpha=0.05, mu=1.0):
        """
        Coverage Width Based Criterion (CWC)
        Combines coverage and width with a penalty term
        """
        coverage = TimeSeriesMetrics.prediction_interval_coverage(y_true, y_pred_lower, y_pred_upper)
        width = TimeSeriesMetrics.prediction_interval_width(y_pred_lower, y_pred_upper)
        
        # Penalty term for insufficient coverage
        target_coverage = 1 - alpha
        penalty = np.exp(-mu * (coverage - target_coverage)) if coverage < target_coverage else 0
        
        return width * (1 + penalty)
    
    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================
    
    @staticmethod
    def diebold_mariano_test(errors_a, errors_b, h=1, power=1):
        """
        Diebold-Mariano test for forecast accuracy comparison
        Tests whether two forecasts have equal accuracy
        
        Parameters:
        - errors_a, errors_b: forecast errors from two models
        - h: forecast horizon
        - power: 1 for MAE, 2 for MSE
        """
        errors_a = np.asarray(errors_a)
        errors_b = np.asarray(errors_b)
        
        if errors_a.shape != errors_b.shape:
            raise ValueError("Error arrays must have the same shape")
        
        n = len(errors_a)
        
        # Calculate loss differential
        if power == 1:
            d = np.abs(errors_a) - np.abs(errors_b)  # MAE
        elif power == 2:
            d = errors_a ** 2 - errors_b ** 2  # MSE
        else:
            raise ValueError("Power must be 1 (MAE) or 2 (MSE)")
        
        # Calculate mean loss differential
        d_bar = np.mean(d)
        
        # Calculate autocorrelation-corrected variance
        # Using Bartlett kernel for autocovariance estimation
        gamma_0 = np.var(d, ddof=1)
        
        # Estimate autocovariances
        max_lag = min(h - 1, n - 1)
        autocovariances = []
        
        for lag in range(1, max_lag + 1):
            if lag < n:
                autocov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
                autocovariances.append(autocov)
        
        # Bartlett kernel weights
        weights = [1 - lag/(max_lag + 1) for lag in range(1, max_lag + 1)]
        
        # Long-run variance
        long_run_var = gamma_0 + 2 * sum(w * gamma for w, gamma in zip(weights, autocovariances))
        
        if long_run_var <= 0:
            long_run_var = gamma_0  # Fallback to simple variance
        
        # DM test statistic
        dm_stat = d_bar / np.sqrt(long_run_var / n)
        
        # p-value (two-sided test)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        return {
            'dm_statistic': dm_stat,
            'p_value': p_value,
            'mean_difference': d_bar,
            'long_run_variance': long_run_var
        }
    
    @staticmethod
    def harvey_leybourne_newbold_test(errors_a, errors_b, h=1):
        """
        Harvey, Leybourne, and Newbold test
        Modified DM test with small sample correction
        """
        dm_result = TimeSeriesMetrics.diebold_mariano_test(errors_a, errors_b, h, power=2)
        
        n = len(errors_a)
        hl_stat = dm_result['dm_statistic'] * np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        
        # p-value from t-distribution
        p_value = 2 * (1 - stats.t.cdf(np.abs(hl_stat), df=n-1))
        
        return {
            'hl_statistic': hl_stat,
            'p_value': p_value,
            'mean_difference': dm_result['mean_difference']
        }
    
    # =========================================================================
    # TIME SERIES SPECIFIC METRICS
    # =========================================================================
    
    @staticmethod
    def mean_absolute_scaled_error_rolling(y_true, y_pred, y_train, window_size=10):
        """
        Rolling MASE - MASE calculated over rolling windows
        Useful for assessing performance stability over time
        """
        y_true, y_pred, y_train = TimeSeriesMetrics._validate_inputs(y_true, y_pred, y_train)
        
        n = len(y_true)
        if n < window_size:
            return TimeSeriesMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train)
        
        mase_values = []
        for i in range(n - window_size + 1):
            window_true = y_true[i:i+window_size]
            window_pred = y_pred[i:i+window_size]
            
            # Use expanding window of training data
            train_window = np.concatenate([y_train, y_true[:i]]) if i > 0 else y_train
            
            try:
                mase_window = TimeSeriesMetrics.mean_absolute_scaled_error(
                    window_true, window_pred, train_window
                )
                mase_values.append(mase_window)
            except:
                continue
        
        return np.mean(mase_values) if mase_values else float('inf')
    
    @staticmethod
    def scaled_errors_distribution(y_true, y_pred, y_train):
        """
        Analyze distribution of scaled errors
        Returns mean, std, and normality test of scaled errors
        """
        y_true, y_pred, y_train = TimeSeriesMetrics._validate_inputs(y_true, y_pred, y_train)
        
        # Calculate naive errors from training data
        naive_errors = np.abs(np.diff(y_train))
        scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        
        if scale == 0:
            return {'mean': float('inf'), 'std': float('inf'), 'normality_pvalue': 0.0}
        
        # Calculate scaled errors
        scaled_errors = np.abs(y_true - y_pred) / scale
        
        # Analyze distribution
        normality_test = stats.normaltest(scaled_errors)
        
        return {
            'mean': np.mean(scaled_errors),
            'std': np.std(scaled_errors),
            'skewness': stats.skew(scaled_errors),
            'kurtosis': stats.kurtosis(scaled_errors),
            'normality_pvalue': normality_test.pvalue,
            'is_normal': normality_test.pvalue > 0.05
        }
    
    @staticmethod
    def forecast_bias(y_true, y_pred):
        """Mean forecast error (bias)"""
        y_true, y_pred, _ = TimeSeriesMetrics._validate_inputs(y_true, y_pred)
        return np.mean(y_true - y_pred)
    
    @staticmethod
    def forecast_accuracy_ratio(y_true, y_pred, y_train):
        """Ratio of forecast accuracy to naive forecast accuracy"""
        mase = TimeSeriesMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train)
        return 1 / mase if mase != 0 else float('inf')
    
    # =========================================================================
    # MULTI-HORIZON METRICS
    # =========================================================================
    
    @staticmethod
    def mase_by_horizon(y_true, y_pred, y_train, max_horizon=None):
        """
        Calculate MASE for different forecast horizons
        Useful for multi-step ahead forecasting
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_train = np.asarray(y_train)
        
        if y_true.ndim == 1:
            # Single series
            if max_horizon is None:
                max_horizon = len(y_true)
            
            mase_by_h = []
            for h in range(1, max_horizon + 1):
                if h <= len(y_true):
                    try:
                        mase_h = TimeSeriesMetrics.mean_absolute_scaled_error(
                            y_true[:h], y_pred[:h], y_train
                        )
                        mase_by_h.append(mase_h)
                    except:
                        mase_by_h.append(float('inf'))
                else:
                    break
            
            return mase_by_h
        
        else:
            # Multiple horizons (matrix: horizons x time)
            n_horizons = y_true.shape[0]
            mase_by_h = []
            
            for h in range(n_horizons):
                try:
                    mase_h = TimeSeriesMetrics.mean_absolute_scaled_error(
                        y_true[h], y_pred[h], y_train
                    )
                    mase_by_h.append(mase_h)
                except:
                    mase_by_h.append(float('inf'))
            
            return mase_by_h
    
    @staticmethod
    def rmsse_by_horizon(y_true, y_pred, y_train, max_horizon=None):
        """Calculate RMSSE for different forecast horizons"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_train = np.asarray(y_train)
        
        if y_true.ndim == 1:
            if max_horizon is None:
                max_horizon = len(y_true)
            
            rmsse_by_h = []
            for h in range(1, max_horizon + 1):
                if h <= len(y_true):
                    try:
                        rmsse_h = TimeSeriesMetrics.root_mean_squared_scaled_error(
                            y_true[:h], y_pred[:h], y_train
                        )
                        rmsse_by_h.append(rmsse_h)
                    except:
                        rmsse_by_h.append(float('inf'))
                else:
                    break
            
            return rmsse_by_h
        else:
            n_horizons = y_true.shape[0]
            rmsse_by_h = []
            
            for h in range(n_horizons):
                try:
                    rmsse_h = TimeSeriesMetrics.root_mean_squared_scaled_error(
                        y_true[h], y_pred[h], y_train
                    )
                    rmsse_by_h.append(rmsse_h)
                except:
                    rmsse_by_h.append(float('inf'))
            
            return rmsse_by_h
    
    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================
    
    @staticmethod
    def comprehensive_timeseries_report(y_true, y_pred, y_train, 
                                      y_pred_lower=None, y_pred_upper=None,
                                      seasonal_period=1, h=1):
        """
        Comprehensive time series forecasting evaluation report
        """
        report = {}
        
        # Scaled error metrics
        report['scaled_errors'] = {
            'mase': TimeSeriesMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train, seasonal_period),
            'rmsse': TimeSeriesMetrics.root_mean_squared_scaled_error(y_true, y_pred, y_train, seasonal_period),
            'smape': TimeSeriesMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'mape': TimeSeriesMetrics.mean_absolute_percentage_error(y_true, y_pred)
        }
        
        # Absolute and squared errors
        report['absolute_errors'] = {
            'mae': np.mean(np.abs(y_true - y_pred)),
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'max_error': np.max(np.abs(y_true - y_pred))
        }
        
        # Bias and accuracy
        report['bias_accuracy'] = {
            'forecast_bias': TimeSeriesMetrics.forecast_bias(y_true, y_pred),
            'accuracy_ratio': TimeSeriesMetrics.forecast_accuracy_ratio(y_true, y_pred, y_train)
        }
        
        # Prediction interval metrics (if available)
        if y_pred_lower is not None and y_pred_upper is not None:
            report['prediction_intervals'] = {
                'coverage': TimeSeriesMetrics.prediction_interval_coverage(y_true, y_pred_lower, y_pred_upper),
                'average_width': TimeSeriesMetrics.prediction_interval_width(y_pred_lower, y_pred_upper),
                'interval_score': TimeSeriesMetrics.interval_score(y_true, y_pred_lower, y_pred_upper),
                'cwc': TimeSeriesMetrics.coverage_width_based_criterion(y_true, y_pred_lower, y_pred_upper)
            }
        
        # Scaled errors distribution analysis
        report['errors_analysis'] = TimeSeriesMetrics.scaled_errors_distribution(y_true, y_pred, y_train)
        
        # Multi-horizon analysis (if applicable)
        if len(y_true) > 1:
            report['multi_horizon'] = {
                'mase_by_horizon': TimeSeriesMetrics.mase_by_horizon(y_true, y_pred, y_train),
                'rmsse_by_horizon': TimeSeriesMetrics.rmsse_by_horizon(y_true, y_pred, y_train)
            }
        
        return report
    
    @staticmethod
    def model_comparison_report(errors_dict, y_train, h=1, alpha=0.05):
        """
        Compare multiple forecasting models using statistical tests
        """
        models = list(errors_dict.keys())
        n_models = len(models)
        
        if n_models < 2:
            raise ValueError("At least two models required for comparison")
        
        report = {}
        
        # Calculate MASE for each model
        mase_scores = {}
        for model_name, errors in errors_dict.items():
            # Reconstruct y_true and y_pred from errors (assuming y_true is same for all)
            # This is simplified - in practice you'd need true values
            y_pred_dummy = np.zeros_like(errors)  # Placeholder
            y_true_dummy = errors + y_pred_dummy  # Placeholder
            
            try:
                mase = TimeSeriesMetrics.mean_absolute_scaled_error(
                    y_true_dummy, y_pred_dummy, y_train
                )
                mase_scores[model_name] = mase
            except:
                mase_scores[model_name] = float('inf')
        
        report['mase_scores'] = mase_scores
        
        # Pairwise Diebold-Mariano tests
        dm_results = {}
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_a = models[i]
                model_b = models[j]
                
                errors_a = errors_dict[model_a]
                errors_b = errors_dict[model_b]
                
                dm_test = TimeSeriesMetrics.diebold_mariano_test(errors_a, errors_b, h=h)
                hl_test = TimeSeriesMetrics.harvey_leybourne_newbold_test(errors_a, errors_b, h=h)
                
                key = f"{model_a}_vs_{model_b}"
                dm_results[key] = {
                    'dm_statistic': dm_test['dm_statistic'],
                    'dm_pvalue': dm_test['p_value'],
                    'hl_statistic': hl_test['hl_statistic'],
                    'hl_pvalue': hl_test['p_value'],
                    'mean_difference': dm_test['mean_difference'],
                    'significant_dm': dm_test['p_value'] < alpha,
                    'significant_hl': hl_test['p_value'] < alpha
                }
        
        report['pairwise_tests'] = dm_results
        
        # Ranking based on MASE
        sorted_models = sorted(mase_scores.items(), key=lambda x: x[1])
        report['model_ranking'] = sorted_models
        report['best_model'] = sorted_models[0][0] if sorted_models else None
        
        return report
    
    @staticmethod
    def get_core_timeseries_metrics(y_true, y_pred, y_train):
        """Get core time series metrics for quick evaluation"""
        return {
            'mase': TimeSeriesMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train),
            'smape': TimeSeriesMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'bias': TimeSeriesMetrics.forecast_bias(y_true, y_pred)
        }