import pandas as pd
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.analyzer import load_experimental_data, perform_comprehensive_analysis
from src.visualizer import generate_all_curve_fitting_charts

def generate_report(analysis_results, output_file='output/curve_fitting_report.txt'):
    os.makedirs('output', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Curve Fitting Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TEMPERATURE-REACTION RATE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        temp_results = analysis_results['temperature_analysis']
        for exp_id, data in temp_results.items():
            f.write(f"\n{exp_id}:\n")
            f.write(f"  Arrhenius Fit R²: {data['arrhenius']['r_squared']:.4f}\n")
            f.write(f"  Activation Energy: {data['arrhenius']['activation_energy']:.2f} J/mol\n")
            f.write(f"  Exponential Fit R²: {data['exponential']['r_squared']:.4f}\n")
        
        f.write("\n\nPRESSURE-REACTION RATE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        pressure_results = analysis_results['pressure_analysis']
        for temp_key, data in pressure_results.items():
            f.write(f"\n{temp_key}:\n")
            f.write(f"  Linear R²: {data['linear']['r_squared']:.4f}\n")
            f.write(f"  Power R²: {data['power']['r_squared']:.4f}\n")
        
        f.write("\n\nTIME-CONCENTRATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        time_results = analysis_results['time_analysis']
        for exp_id, data in time_results.items():
            f.write(f"\n{exp_id}:\n")
            f.write(f"  Power Fit R²: {data['power']['r_squared']:.4f}\n")
            f.write(f"  Power Exponent: {data['power']['b']:.4f}\n")
    
    print(f"Report saved to {output_file}")

def main():
    print("Curve Fitting Analysis System v1.0")
    print("=" * 40)
    
    os.makedirs('output', exist_ok=True)
    
    df = load_experimental_data('data/experimental_data.csv')
    print(f"Loaded {len(df)} records from {df['experiment_id'].nunique()} experiments")
    print(f"Experiments: {df['experiment_id'].unique().tolist()}")
    print(f"Temperature range: {df['temperature'].min()}°C to {df['temperature'].max()}°C")
    print(f"Pressure range: {df['pressure'].min()} to {df['pressure'].max()} atm")
    
    print("\nPerforming curve fitting analysis...")
    results = perform_comprehensive_analysis(df)
    
    print("\nTemperature Analysis Results:")
    for exp_id, data in results['temperature_analysis'].items():
        print(f"  {exp_id}: Exponential R² = {data['exponential']['r_squared']:.4f}")
    
    print("\nGenerating charts...")
    generate_all_curve_fitting_charts(results)
    
    print("\nGenerating report...")
    generate_report(results)
    
    print("\nAnalysis complete!")
    print("Output files saved to 'output/' directory")

if __name__ == '__main__':
    main()
