import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import warnings

class RegressionMetrics:
    """Comprehensive regression metrics without sklearn"""
    
    @staticmethod
    def _validate_inputs(y_true, y_pred):
        """Validate and convert inputs to numpy arrays"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match")
        
        return y_true, y_pred
    
    @staticmethod
    def _handle_zeros(y_true, y_pred, replace_value=1e-10):
        """Handle zeros for percentage-based metrics"""
        y_true_safe = np.where(y_true == 0, replace_value, y_true)
        y_pred_safe = np.where(y_pred == 0, replace_value, y_pred)
        return y_true_safe, y_pred_safe
    
    # =========================================================================
    # ABSOLUTE ERRORS
    # =========================================================================
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """Mean Absolute Error (MAE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def median_absolute_error(y_true, y_pred):
        """Median Absolute Error (MedAE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        return np.median(np.abs(y_true - y_pred))
    
    @staticmethod
    def max_error(y_true, y_pred):
        """Max Error - Worst case error"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        return np.max(np.abs(y_true - y_pred))
    
    @staticmethod
    def mean_absolute_deviation(y_true, y_pred):
        """Mean Absolute Deviation around predictions"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
    # =========================================================================
    # SQUARED ERRORS
    # =========================================================================
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Mean Squared Error (MSE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """Root Mean Squared Error (RMSE)"""
        return np.sqrt(RegressionMetrics.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_squared_log_error(y_true, y_pred):
        """Mean Squared Logarithmic Error (MSLE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        y_true, y_pred = RegressionMetrics._handle_zeros(y_true, y_pred)
        return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    
    @staticmethod
    def root_mean_squared_log_error(y_true, y_pred):
        """Root Mean Squared Logarithmic Error (RMSLE)"""
        return np.sqrt(RegressionMetrics.mean_squared_log_error(y_true, y_pred))
    
    # =========================================================================
    # PERCENTAGE ERRORS
    # =========================================================================
    
    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        """Mean Absolute Percentage Error (MAPE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        y_true_safe, y_pred_safe = RegressionMetrics._handle_zeros(y_true, y_pred)
        return 100 * np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe))
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error (sMAPE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        y_true_safe, y_pred_safe = RegressionMetrics._handle_zeros(y_true, y_pred)
        denominator = (np.abs(y_true_safe) + np.abs(y_pred_safe)) / 2
        return 100 * np.mean(np.abs(y_true_safe - y_pred_safe) / denominator)
    
    @staticmethod
    def mean_absolute_scaled_error(y_true, y_pred, y_train=None):
        """Mean Absolute Scaled Error (MASE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        
        mae = RegressionMetrics.mean_absolute_error(y_true, y_pred)
        
        if y_train is not None:
            y_train = np.asarray(y_train)
            # Naive forecast (previous value)
            naive_errors = np.abs(np.diff(y_train))
        else:
            # Use y_true if no training data provided
            naive_errors = np.abs(np.diff(y_true))
        
        if len(naive_errors) == 0:
            return float('inf')
        
        scale = np.mean(naive_errors)
        return mae / scale if scale > 0 else float('inf')
    
    @staticmethod
    def mean_percentage_error(y_true, y_pred):
        """Mean Percentage Error (MPE) - Bias direction"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        y_true_safe, y_pred_safe = RegressionMetrics._handle_zeros(y_true, y_pred)
        return 100 * np.mean((y_true_safe - y_pred_safe) / y_true_safe)
    
    # =========================================================================
    # STATISTICAL METRICS
    # =========================================================================
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """R² Score - Coefficient of Determination"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    @staticmethod
    def adjusted_r2_score(y_true, y_pred, n_features):
        """Adjusted R² Score"""
        r2 = RegressionMetrics.r2_score(y_true, y_pred)
        n = len(y_true)
        if n - n_features - 1 <= 0:
            return r2
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    @staticmethod
    def explained_variance_score(y_true, y_pred):
        """Explained Variance Score"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        numerator = np.var(y_true - y_pred)
        denominator = np.var(y_true)
        return 1 - (numerator / denominator) if denominator != 0 else 0.0
    
    @staticmethod
    def mean_squared_error_variance(y_true, y_pred):
        """MSE decomposed into bias² + variance"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        variance = np.var(y_pred - y_true)
        return {
            'bias_squared': bias ** 2,
            'variance': variance,
            'total_mse': bias ** 2 + variance
        }
    
    # =========================================================================
    # ROBUST METRICS
    # =========================================================================
    
    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        """Huber Loss - Robust to outliers"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        error = np.abs(y_true - y_pred)
        quadratic = np.minimum(error, delta)
        linear = error - quadratic
        return np.mean(0.5 * quadratic ** 2 + delta * linear)
    
    @staticmethod
    def tukey_biweight_loss(y_true, y_pred, c=4.685):
        """Tukey's Biweight Loss - Highly robust to outliers"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        error = y_true - y_pred
        scaled_error = error / c
        
        # Biweight function
        mask = np.abs(scaled_error) < 1
        loss = np.zeros_like(error)
        loss[mask] = (c ** 2 / 6) * (1 - (1 - scaled_error[mask] ** 2) ** 3)
        loss[~mask] = c ** 2 / 6
        
        return np.mean(loss)
    
    @staticmethod
    def log_cosh_loss(y_true, y_pred):
        """Log-Cosh Loss - Smooth approximation of MAE"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        error = y_true - y_pred
        return np.mean(np.log(np.cosh(error)))
    
    @staticmethod
    def quantile_loss(y_true, y_pred, quantile=0.5):
        """Quantile Loss - For quantile regression"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        error = y_true - y_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
    
    # =========================================================================
    # DISTRIBUTION-BASED METRICS
    # =========================================================================
    
    @staticmethod
    def gaussian_negative_log_likelihood(y_true, y_pred, y_std=None):
        """Gaussian Negative Log-Likelihood"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        
        if y_std is None:
            # Estimate std from residuals
            residuals = y_true - y_pred
            y_std = np.std(residuals)
        
        if np.any(y_std <= 0):
            raise ValueError("Standard deviation must be positive")
        
        nll = 0.5 * np.mean((y_true - y_pred) ** 2 / y_std ** 2 + np.log(2 * np.pi * y_std ** 2))
        return nll
    
    @staticmethod
    def gamma_negative_log_likelihood(y_true, y_pred, dispersion=1.0):
        """Gamma Negative Log-Likelihood - For positive-valued data"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        
        if np.any(y_pred <= 0) or np.any(y_true <= 0):
            raise ValueError("Gamma NLL requires positive predictions and true values")
        
        # Using shape-rate parameterization
        shape = 1 / dispersion
        rate = shape / y_pred
        
        nll = np.mean(shape * np.log(rate) - (shape - 1) * np.log(y_true) + rate * y_true - np.log(stats.gamma(shape).pdf(shape)))
        return nll
    
    @staticmethod
    def poisson_negative_log_likelihood(y_true, y_pred):
        """Poisson Negative Log-Likelihood - For count data"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        
        if np.any(y_pred <= 0):
            raise ValueError("Poisson NLL requires positive predictions")
        
        nll = np.mean(y_pred - y_true * np.log(y_pred) + np.log(stats.poisson(y_pred).pmf(y_true)))
        return nll
    
    # =========================================================================
    # QUANTILE METRICS
    # =========================================================================
    
    @staticmethod
    def pinball_loss(y_true, y_pred_lower, y_pred_upper, quantile=0.5):
        """Pinball Loss for quantile regression"""
        y_true = np.asarray(y_true)
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)
        
        loss_lower = RegressionMetrics.quantile_loss(y_true, y_pred_lower, quantile)
        loss_upper = RegressionMetrics.quantile_loss(y_true, y_pred_upper, 1 - quantile)
        
        return (loss_lower + loss_upper) / 2
    
    @staticmethod
    def quantile_coverage(y_true, y_pred_lower, y_pred_upper):
        """Quantile Coverage - Empirical coverage of prediction intervals"""
        y_true = np.asarray(y_true)
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)
        
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        return coverage
    
    @staticmethod
    def interval_width(y_pred_lower, y_pred_upper):
        """Average Prediction Interval Width"""
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)
        
        return np.mean(y_pred_upper - y_pred_lower)
    
    @staticmethod
    def interval_score(y_true, y_pred_lower, y_pred_upper, alpha=0.05):
        """Interval Score - Combines coverage and width"""
        y_true = np.asarray(y_true)
        y_pred_lower = np.asarray(y_pred_lower)
        y_pred_upper = np.asarray(y_pred_upper)
        
        width = y_pred_upper - y_pred_lower
        penalty = (2 / alpha) * np.where(
            y_true < y_pred_lower, y_pred_lower - y_true,
            np.where(y_true > y_pred_upper, y_true - y_pred_upper, 0)
        )
        
        return np.mean(width + penalty)
    
    # =========================================================================
    # TIME SERIES METRICS
    # =========================================================================
    
    @staticmethod
    def mean_absolute_scaled_error_ts(y_true, y_pred, y_train):
        """MASE for time series (explicit training data required)"""
        return RegressionMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train)
    
    @staticmethod
    def root_mean_squared_scaled_error(y_true, y_pred, y_train):
        """Root Mean Squared Scaled Error (RMSSE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        y_train = np.asarray(y_train)
        
        rmse = RegressionMetrics.root_mean_squared_error(y_true, y_pred)
        
        # Seasonal naive errors (assuming period=1 for simplicity)
        naive_errors = np.diff(y_train)
        scale = np.sqrt(np.mean(naive_errors ** 2))
        
        return rmse / scale if scale > 0 else float('inf')
    
    @staticmethod
    def mean_forecast_error(y_true, y_pred):
        """Mean Forecast Error (MFE) - Bias in forecasts"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        return np.mean(y_true - y_pred)
    
    @staticmethod
    def mean_absolute_forecast_error(y_true, y_pred):
        """Mean Absolute Forecast Error (MAFE)"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mean_absolute_percentage_forecast_error(y_true, y_pred):
        """Mean Absolute Percentage Forecast Error"""
        return RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred)
    
    @staticmethod
    def theils_u_statistic(y_true, y_pred):
        """Theil's U Statistic - Compares to naive forecast"""
        y_true, y_pred = RegressionMetrics._validate_inputs(y_true, y_pred)
        
        # Naive forecast (previous period)
        naive_forecast = np.roll(y_true, 1)
        naive_forecast[0] = y_true[0]  # Handle first element
        
        mse_model = RegressionMetrics.mean_squared_error(y_true, y_pred)
        mse_naive = RegressionMetrics.mean_squared_error(y_true, naive_forecast)
        
        return np.sqrt(mse_model / mse_naive) if mse_naive > 0 else float('inf')
    
    # =========================================================================
    # COMPREHENSIVE EVALUATION
    # =========================================================================
    
    @staticmethod
    def comprehensive_regression_report(y_true, y_pred, y_train=None, 
                                      y_pred_lower=None, y_pred_upper=None,
                                      problem_type='standard'):
        """
        Comprehensive regression evaluation report
        
        Parameters:
        - problem_type: 'standard', 'time_series', 'probabilistic'
        """
        report = {}
        
        # Basic absolute errors
        report['absolute_errors'] = {
            'mae': RegressionMetrics.mean_absolute_error(y_true, y_pred),
            'medae': RegressionMetrics.median_absolute_error(y_true, y_pred),
            'max_error': RegressionMetrics.max_error(y_true, y_pred)
        }
        
        # Squared errors
        report['squared_errors'] = {
            'mse': RegressionMetrics.mean_squared_error(y_true, y_pred),
            'rmse': RegressionMetrics.root_mean_squared_error(y_true, y_pred),
            'rmsle': RegressionMetrics.root_mean_squared_log_error(y_true, y_pred)
        }
        
        # Percentage errors
        report['percentage_errors'] = {
            'mape': RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred),
            'smape': RegressionMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'mpe': RegressionMetrics.mean_percentage_error(y_true, y_pred)
        }
        
        # Statistical metrics
        report['statistical_metrics'] = {
            'r2': RegressionMetrics.r2_score(y_true, y_pred),
            'explained_variance': RegressionMetrics.explained_variance_score(y_true, y_pred)
        }
        
        # Robust metrics
        report['robust_metrics'] = {
            'huber_loss': RegressionMetrics.huber_loss(y_true, y_pred),
            'log_cosh_loss': RegressionMetrics.log_cosh_loss(y_true, y_pred)
        }
        
        # Time series specific metrics
        if problem_type == 'time_series' and y_train is not None:
            report['time_series_metrics'] = {
                'mase': RegressionMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train),
                'rmsse': RegressionMetrics.root_mean_squared_scaled_error(y_true, y_pred, y_train),
                'mfe': RegressionMetrics.mean_forecast_error(y_true, y_pred),
                'mafe': RegressionMetrics.mean_absolute_forecast_error(y_true, y_pred),
                'theils_u': RegressionMetrics.theils_u_statistic(y_true, y_pred)
            }
        
        # Probabilistic metrics
        if y_pred_lower is not None and y_pred_upper is not None:
            report['probabilistic_metrics'] = {
                'pinball_loss': RegressionMetrics.pinball_loss(y_true, y_pred_lower, y_pred_upper),
                'quantile_coverage': RegressionMetrics.quantile_coverage(y_true, y_pred_lower, y_pred_upper),
                'interval_width': RegressionMetrics.interval_width(y_pred_lower, y_pred_upper),
                'interval_score': RegressionMetrics.interval_score(y_true, y_pred_lower, y_pred_upper)
            }
        
        # Error distribution analysis
        residuals = y_true - y_pred
        report['error_analysis'] = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skewness_residual': stats.skew(residuals),
            'kurtosis_residual': stats.kurtosis(residuals),
            'normality_test_pvalue': stats.normaltest(residuals).pvalue
        }
        
        return report
    
    @staticmethod
    def get_core_metrics(y_true, y_pred):
        """Get core regression metrics for quick evaluation"""
        return {
            'mae': RegressionMetrics.mean_absolute_error(y_true, y_pred),
            'rmse': RegressionMetrics.root_mean_squared_error(y_true, y_pred),
            'r2': RegressionMetrics.r2_score(y_true, y_pred),
            'mape': RegressionMetrics.mean_absolute_percentage_error(y_true, y_pred)
        }
    
    @staticmethod
    def get_advanced_metrics(y_true, y_pred, y_train=None):
        """Get advanced metrics for comprehensive analysis"""
        core = RegressionMetrics.get_core_metrics(y_true, y_pred)
        advanced = {
            'medae': RegressionMetrics.median_absolute_error(y_true, y_pred),
            'max_error': RegressionMetrics.max_error(y_true, y_pred),
            'rmsle': RegressionMetrics.root_mean_squared_log_error(y_true, y_pred),
            'smape': RegressionMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'explained_variance': RegressionMetrics.explained_variance_score(y_true, y_pred),
            'huber_loss': RegressionMetrics.huber_loss(y_true, y_pred)
        }
        
        if y_train is not None:
            advanced['mase'] = RegressionMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train)
        
        return {**core, **advanced}