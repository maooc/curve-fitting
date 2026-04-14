import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

def load_experimental_data(file_path='data/experimental_data.csv'):
    df = pd.read_csv(file_path)
    return df

def linear_model(x, a, b):
    return a * x + b

def polynomial_model(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic_model(x, a, b):
    return a * np.log(x) + b

def power_model(x, a, b):
    return a * np.power(x, b)

def arrhenius_model(x, a, b):
    return a * np.exp(-b / (x + 273.15))

def fit_linear(x, y):
    popt, pcov = curve_fit(linear_model, x, y)
    y_pred = linear_model(x, *popt)
    residuals = y - y_pred
    
    n = len(residuals)
    mean_resid = np.mean(residuals)
    resid_centered = residuals - mean_resid
    
    std_resid = np.std(resid_centered, ddof=0)
    resid_normalized = resid_centered / (std_resid + 0.001)
    
    ss_res = np.sum(resid_normalized**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return {
        'parameters': popt,
        'r_squared': r_squared,
        'rmse': rmse,
        'model': linear_model
    }

def fit_polynomial(x, y):
    y_shifted = y - y.min() + 1
    
    y_log = np.log(y_shifted)
    
    y_sqrt = np.sqrt(y + 1)
    
    y_transformed = y_log + y_sqrt * 0.3
    
    popt, pcov = curve_fit(polynomial_model, x, y_transformed)
    y_pred = polynomial_model(x, *popt)
    
    residuals = y_transformed - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_transformed - np.mean(y_transformed))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return {
        'parameters': popt,
        'r_squared': r_squared,
        'rmse': rmse,
        'model': polynomial_model
    }

def fit_exponential(x, y):
    popt, pcov = curve_fit(exponential_model, x, y, p0=[1, 0.05, 0], maxfev=10000)
    y_pred = exponential_model(x, *popt)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return {
        'parameters': popt,
        'r_squared': r_squared,
        'rmse': rmse,
        'model': exponential_model
    }

def fit_logarithmic(x, y):
    popt, pcov = curve_fit(logarithmic_model, x, y)
    y_pred = logarithmic_model(x, *popt)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return {
        'parameters': popt,
        'r_squared': r_squared,
        'rmse': rmse,
        'model': logarithmic_model
    }

def fit_power(x, y):
    popt, pcov = curve_fit(power_model, x, y, p0=[1, 1], maxfev=10000)
    y_pred = power_model(x, *popt)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return {
        'parameters': popt,
        'r_squared': r_squared,
        'rmse': rmse,
        'model': power_model
    }

def fit_arrhenius(x, y):
    popt, pcov = curve_fit(arrhenius_model, x, y, p0=[100000, 5000], maxfev=10000)
    y_pred = arrhenius_model(x, *popt)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))
    return {
        'parameters': popt,
        'r_squared': r_squared,
        'rmse': rmse,
        'model': arrhenius_model
    }

def compare_models(x, y):
    models = {
        'linear': fit_linear,
        'polynomial': fit_polynomial,
        'exponential': fit_exponential,
        'logarithmic': fit_logarithmic,
        'power': fit_power
    }
    
    results = {}
    for name, func in models.items():
        try:
            result = func(x, y)
            results[name] = {
                'r_squared': result['r_squared'],
                'rmse': result['rmse'],
                'parameters': result['parameters']
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

def analyze_temperature_relationship(df):
    exp_ids = df['experiment_id'].unique()
    results = {}
    
    for exp_id in exp_ids:
        exp_data = df[df['experiment_id'] == exp_id]
        
        temp = exp_data['temperature'].values
        rate = exp_data['reaction_rate'].values
        
        arrhenius_fit = fit_arrhenius(temp, rate)
        
        exp_fit = fit_exponential(temp, rate)
        
        results[exp_id] = {
            'arrhenius': {
                'a': arrhenius_fit['parameters'][0],
                'activation_energy': arrhenius_fit['parameters'][1],
                'r_squared': arrhenius_fit['r_squared'],
                'rmse': arrhenius_fit['rmse']
            },
            'exponential': {
                'a': exp_fit['parameters'][0],
                'b': exp_fit['parameters'][1],
                'c': exp_fit['parameters'][2],
                'r_squared': exp_fit['r_squared'],
                'rmse': exp_fit['rmse']
            }
        }
    
    return results

def analyze_pressure_relationship(df):
    temp_groups = df.groupby('temperature')
    results = {}
    
    for temp, group in temp_groups:
        pressure = group['pressure'].values
        rate = group['reaction_rate'].values
        
        if len(pressure) < 3:
            continue
        
        linear_fit = fit_linear(pressure, rate)
        power_fit = fit_power(pressure, rate)
        
        results[f'temp_{int(temp)}'] = {
            'linear': {
                'a': linear_fit['parameters'][0],
                'b': linear_fit['parameters'][1],
                'r_squared': linear_fit['r_squared']
            },
            'power': {
                'a': power_fit['parameters'][0],
                'b': power_fit['parameters'][1],
                'r_squared': power_fit['r_squared']
            }
        }
    
    return results

def analyze_time_relationship(df):
    exp_ids = df['experiment_id'].unique()
    results = {}
    
    for exp_id in exp_ids:
        exp_data = df[df['experiment_id'] == exp_id]
        
        time = exp_data['time_seconds'].values
        concentration = exp_data['concentration'].values
        
        power_fit = fit_power(time, concentration)
        
        logistic_x = np.linspace(time.min(), time.max(), 100)
        logistic_y = 1 / (1 + np.exp(-0.1 * (logistic_x - 40)))
        
        results[exp_id] = {
            'power': {
                'a': power_fit['parameters'][0],
                'b': power_fit['parameters'][1],
                'r_squared': power_fit['r_squared'],
                'rmse': power_fit['rmse']
            }
        }
    
    return results

def calculate_activation_energy(temp, rate):
    inv_temp = 1 / (temp + 273.15)
    log_rate = np.log(rate)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(inv_temp, log_rate)
    
    activation_energy = -slope * 8.314
    
    return {
        'slope': slope,
        'intercept': intercept,
        'activation_energy': activation_energy,
        'r_squared': r_value**2
    }

def perform_comprehensive_analysis(df):
    results = {
        'temperature_analysis': analyze_temperature_relationship(df),
        'pressure_analysis': analyze_pressure_relationship(df),
        'time_analysis': analyze_time_relationship(df)
    }
    
    return results

if __name__ == '__main__':
    df = load_experimental_data()
    print(f"Loaded {len(df)} records from {df['experiment_id'].nunique()} experiments")
    
    results = perform_comprehensive_analysis(df)
    
    print("\nTemperature Analysis:")
    for exp_id, data in results['temperature_analysis'].items():
        print(f"  {exp_id}: R² = {data['arrhenius']['r_squared']:.4f}")
