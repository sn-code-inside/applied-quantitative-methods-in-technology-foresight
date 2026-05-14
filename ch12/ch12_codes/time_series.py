"""
Time Series Analysis and Forecasting Module
Chapter 12: Data-Led Technology Roadmapping

This module provides functions for temporal trend analysis,
S-curve fitting, and ARIMA forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA


def analyze_temporal_trends(df: pd.DataFrame,
                            date_column: str = 'Year') -> tuple:
    """
    Analyze and visualize temporal trends in technology activity.
    
    Parameters:
        df: DataFrame with date column
        date_column: Name of the date column
        
    Returns:
        yearly_counts: Series of document counts by year
        growth_rate: Mean annual growth rate (percentage)
    """
    # Extract year from datetime column
    df = df.copy()
    df['year_int'] = df[date_column].dt.year
    
    # Count publications/patents per year
    yearly_counts = df.groupby('year_int').size()
    
    # Calculate growth rate
    growth_rate = yearly_counts.pct_change().mean() * 100
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_counts.plot(kind='line', marker='o', linewidth=2,
                       markersize=8, ax=ax, color='steelblue')
    
    # Add trend line
    z = np.polyfit(yearly_counts.index, yearly_counts.values, 1)
    p = np.poly1d(z)
    ax.plot(yearly_counts.index, p(yearly_counts.index),
            '--', color='red', alpha=0.7, label='Linear Trend')
    
    ax.set_title(f'Technology Activity Over Time (Avg Growth: {growth_rate:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temporal_trends.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return yearly_counts, growth_rate


def logistic_growth(t: np.ndarray, K: float, P0: float, r: float) -> np.ndarray:
    """
    Logistic growth model (S-curve).
    
    Parameters:
        t: Time points
        K: Carrying capacity (asymptote)
        P0: Initial value
        r: Growth rate
        
    Returns:
        Predicted values at each time point
    """
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))


def fit_scurve(yearly_data: pd.Series) -> dict:
    """
    Fit S-curve to cumulative technology data.
    
    Parameters:
        yearly_data: Series with year index and document counts
        
    Returns:
        Dictionary with fitted parameters (K, P0, r) and predictions
    """
    years = yearly_data.index.values
    cumulative = yearly_data.cumsum().values
    
    # Normalize time to start at 0
    t = years - years.min()
    
    # Initial parameter estimates
    K_est = cumulative[-1] * 1.5  # Carrying capacity estimate
    P0_est = max(cumulative[0], 1)  # Initial value (avoid zero)
    r_est = 0.3  # Growth rate estimate
    
    # Fit curve
    try:
        params, covariance = curve_fit(
            logistic_growth, t, cumulative,
            p0=[K_est, P0_est, r_est],
            maxfev=10000,
            bounds=([0, 0, 0], [np.inf, np.inf, 10])
        )
        K, P0, r = params
        
        # Generate predictions (extend 10 years)
        t_extended = np.arange(0, len(t) + 10)
        predictions = logistic_growth(t_extended, K, P0, r)
        
        # Calculate inflection point (where growth rate peaks)
        inflection_t = np.log((K - P0) / P0) / r
        inflection_year = years.min() + inflection_t
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(years, cumulative, label='Observed', s=60, color='steelblue')
        
        extended_years = np.arange(years.min(), years.min() + len(t_extended))
        ax.plot(extended_years, predictions, 'r-', linewidth=2, 
                label=f'S-Curve Fit (K={K:.0f})')
        ax.axvline(x=inflection_year, color='green', linestyle='--', 
                   label=f'Inflection Point ({inflection_year:.0f})')
        
        ax.set_title('Technology S-Curve Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cumulative Documents', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('scurve_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'K': K,
            'P0': P0,
            'r': r,
            'inflection_year': inflection_year,
            'predictions': predictions,
            'years_extended': extended_years
        }
        
    except Exception as e:
        print(f"S-curve fitting failed: {e}")
        return None


def forecast_technology_trend(yearly_counts: pd.Series,
                              periods: int = 5) -> tuple:
    """
    Forecast future technology activity using ARIMA.
    
    Parameters:
        yearly_counts: Historical time series data
        periods: Number of years to forecast
        
    Returns:
        forecast: Forecasted values
        fitted_model: Fitted ARIMA model
    """
    # Fit ARIMA(1,1,1) model
    model = ARIMA(yearly_counts, order=(1, 1, 1))
    fitted_model = model.fit()
    
    # Generate forecast with confidence intervals
    forecast_result = fitted_model.get_forecast(steps=periods)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(yearly_counts.index, yearly_counts.values,
            'o-', label='Historical', linewidth=2, color='steelblue')
    
    # Plot forecast
    forecast_years = range(yearly_counts.index[-1] + 1,
                          yearly_counts.index[-1] + periods + 1)
    ax.plot(forecast_years, forecast.values, 's--',
            label='Forecast', color='red', linewidth=2)
    
    # Plot confidence interval
    ax.fill_between(forecast_years, 
                    conf_int.iloc[:, 0], 
                    conf_int.iloc[:, 1],
                    color='red', alpha=0.2, label='95% Confidence Interval')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.set_title('Technology Trend Forecast (ARIMA)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('arima_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nARIMA Forecast Summary:")
    print(f"Historical period: {yearly_counts.index.min()}-{yearly_counts.index.max()}")
    print(f"Forecast period: {list(forecast_years)[0]}-{list(forecast_years)[-1]}")
    print(f"Forecasted values: {forecast.values.round(0)}")
    
    return forecast, fitted_model


def assess_technology_lifecycle(yearly_data: pd.Series) -> str:
    """
    Assess technology lifecycle stage based on growth patterns.
    
    Parameters:
        yearly_data: Series with year index and document counts
        
    Returns:
        Lifecycle stage: 'Emerging', 'Growth', 'Mature', or 'Declining'
    """
    # Calculate recent growth
    recent_growth = yearly_data.tail(3).pct_change().mean() * 100
    overall_growth = yearly_data.pct_change().mean() * 100
    
    # Calculate acceleration (change in growth rate)
    growth_rates = yearly_data.pct_change()
    acceleration = growth_rates.diff().tail(3).mean() * 100
    
    # Determine lifecycle stage
    if recent_growth > 30 and acceleration > 0:
        stage = 'Emerging'
    elif recent_growth > 15:
        stage = 'Growth'
    elif recent_growth > 0:
        stage = 'Mature'
    else:
        stage = 'Declining'
    
    print(f"\nTechnology Lifecycle Assessment:")
    print(f"  Recent growth rate: {recent_growth:.1f}%")
    print(f"  Overall growth rate: {overall_growth:.1f}%")
    print(f"  Growth acceleration: {acceleration:.1f}%")
    print(f"  Lifecycle stage: {stage}")
    
    return stage


# Example usage
if __name__ == "__main__":
    print("Time Series Module - Chapter 12")
    print("Functions: analyze_temporal_trends(), fit_scurve(), forecast_technology_trend()")
    print("See README.md for complete documentation")
