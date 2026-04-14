import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_model(x, a, b):
    return a * x + b

def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

def power_model(x, a, b):
    return a * np.power(x, b)

def plot_temperature_rate_fit(exp_data, fit_results, save_path='output/temperature_rate_fit.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    temp = exp_data['temperature'].values
    rate = exp_data['reaction_rate'].values
    
    temp_smooth = np.linspace(temp.min(), temp.max(), 100)
    
    exp_fit = fit_results['exponential']
    y_exp = exponential_model(temp_smooth, exp_fit['a'], exp_fit['b'], exp_fit['c'])
    
    ax1.scatter(temp, rate, color='blue', s=50, label='Experimental Data', zorder=5)
    ax1.plot(temp_smooth, y_exp, 'r-', linewidth=2, label=f'Exponential Fit (R²={exp_fit["r_squared"]:.4f})')
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Reaction Rate', fontsize=12)
    ax1.set_title('Temperature vs Reaction Rate', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    inv_temp = 1 / (temp + 273.15)
    log_rate = np.log(rate)
    ax2.scatter(inv_temp * 1000, log_rate, color='green', s=50, label='Data', zorder=5)
    
    z = np.polyfit(inv_temp, log_rate, 1)
    p = np.poly1d(z)
    ax2.plot(inv_temp * 1000, p(inv_temp), 'r--', linewidth=2, label='Linear Fit')
    ax2.set_xlabel('1000/T (K⁻¹)', fontsize=12)
    ax2.set_ylabel('ln(Rate)', fontsize=12)
    ax2.set_title('Arrhenius Plot', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved temperature-rate fit to {save_path}")

def plot_pressure_rate_fit(pressure_data, save_path='output/pressure_rate_fit.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pressure = pressure_data['pressure'].values
    rate = pressure_data['reaction_rate'].values
    
    z = np.polyfit(pressure, rate, 1)
    p = np.poly1d(z)
    pressure_smooth = np.linspace(pressure.min(), pressure.max(), 100)
    
    ax.scatter(pressure, rate, color='blue', s=80, label='Experimental Data', zorder=5)
    ax.plot(pressure_smooth, p(pressure_smooth), 'r-', linewidth=2, label='Linear Fit')
    
    for i, (p, r) in enumerate(zip(pressure, rate)):
        ax.annotate(f'{int(rate[i])}', (p, r), textcoords='offset points', xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('Pressure (atm)', fontsize=12)
    ax.set_ylabel('Reaction Rate', fontsize=12)
    ax.set_title('Pressure vs Reaction Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pressure-rate fit to {save_path}")

def plot_time_concentration_fit(exp_data, save_path='output/time_concentration_fit.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = exp_data['time_seconds'].values
    conc = exp_data['concentration'].values
    
    z = np.polyfit(time, conc, 2)
    p = np.poly1d(z)
    time_smooth = np.linspace(time.min(), time.max(), 100)
    
    ax.scatter(time, conc, color='blue', s=80, label='Experimental Data', zorder=5)
    ax.plot(time_smooth, p(time_smooth), 'r-', linewidth=2, label='Polynomial Fit')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Concentration', fontsize=12)
    ax.set_title('Time vs Concentration', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved time-concentration fit to {save_path}")

def plot_yield_comparison(df, save_path='output/yield_comparison.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = df['experiment_id'].unique()
    temperatures = sorted(df['temperature'].unique())
    
    x = np.arange(len(temperatures))
    width = 0.15
    
    for i, exp_id in enumerate(experiments):
        exp_data = df[df['experiment_id'] == exp_id]
        yields = [exp_data[exp_data['temperature'] == temp]['yield_percent'].values[0] 
                 for temp in temperatures]
        
        offset = (i - len(experiments)/2) * width
        bars = ax.bar(x + offset, yields, width, label=exp_id, edgecolor='black')
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Yield (%)', fontsize=12)
    ax.set_title('Yield Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(temperatures)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved yield comparison to {save_path}")

def plot_model_comparison(x, y, model_results, save_path='output/model_comparison.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(x, y, color='blue', s=80, label='Experimental Data', zorder=5)
    
    x_smooth = np.linspace(x.min(), x.max(), 100)
    
    colors = ['red', 'green', 'purple', 'orange']
    labels = list(model_results.keys())
    
    for i, (name, result) in enumerate(model_results.items()):
        if 'error' in result:
            continue
        
        params = result['parameters']
        r_sq = result['r_squared']
        
        if name == 'linear':
            y_pred = linear_model(x_smooth, *params)
        elif name == 'exponential':
            y_pred = exponential_model(x_smooth, *params)
        elif name == 'power':
            y_pred = power_model(x_smooth, *params)
        else:
            continue
        
        ax.plot(x_smooth, y_pred, color=colors[i % len(colors)], linewidth=2, 
                label=f'{name} (R²={r_sq:.4f})')
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Reaction Rate', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison to {save_path}")

def plot_residuals(x, y, y_pred, model_name, save_path='output/residuals.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y - y_pred
    
    ax.scatter(x, residuals, color='red', s=50, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(f'Residual Plot - {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved residuals to {save_path}")

def generate_all_curve_fitting_charts(analysis_results):
    import os
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(script_dir))
    os.makedirs('output', exist_ok=True)
    
    from src.analyzer import load_experimental_data, compare_models
    df = load_experimental_data()
    
    for exp_id in df['experiment_id'].unique():
        exp_data = df[df['experiment_id'] == exp_id]
        
        temp = exp_data['temperature'].values
        rate = exp_data['reaction_rate'].values
        
        model_results = compare_models(temp, rate)
        
        try:
            plot_temperature_rate_fit(exp_data, analysis_results['temperature_analysis'][exp_id], 
                                     f'output/temp_rate_{exp_id}.png')
            plot_model_comparison(temp, rate, model_results, f'output/models_{exp_id}.png')
        except Exception as e:
            print(f"Error plotting for {exp_id}: {e}")
        
        try:
            plot_time_concentration_fit(exp_data, f'output/time_conc_{exp_id}.png')
        except Exception as e:
            print(f"Error plotting time-concentration for {exp_id}: {e}")
    
    try:
        plot_yield_comparison(df)
    except Exception as e:
        print(f"Error plotting yield comparison: {e}")
    
    temp_groups = df.groupby('temperature')
    for temp, group in temp_groups:
        try:
            plot_pressure_rate_fit(group, f'output/pressure_temp_{int(temp)}.png')
        except Exception as e:
            print(f"Error plotting pressure for temp {temp}: {e}")

if __name__ == '__main__':
    from analyzer import load_experimental_data, perform_comprehensive_analysis
    
    df = load_experimental_data()
    results = perform_comprehensive_analysis(df)
    generate_all_curve_fitting_charts(results)
    print("All curve fitting charts generated!")
