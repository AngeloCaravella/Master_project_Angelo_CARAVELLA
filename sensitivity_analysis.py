
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
import itertools

# Import necessary functions from your existing project
from run_experiments import run_benchmark, get_algorithms, calculate_max_cs
from ev2gym.rl_agent import reward as reward_module

# --- Helper Functions ---

def get_user_input(prompt, default=None):
    """Gets user input from the console with an optional default value."""
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input or default

def select_from_list(items, prompt):
    """Displays a list of items and asks the user to select one."""
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"  {i+1}. {item}")
    choice = int(get_user_input("Enter the number of your choice", 1)) - 1
    return items[choice]

def set_nested_dict_value(d, path, value):
    """Sets a value in a nested dictionary using a list of keys."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value

def run_sensitivity_analysis():
    """Main function to configure and run the sensitivity analysis."""
    
    # --- 1. Define Key Parameters for Analysis ---
    # Maps a user-friendly name to the path of keys in the YAML file.
    key_parameters = {
        "Number of Charging Stations": ['number_of_charging_stations'],
        "Transformer Max Power (kW)": ['transformer', 'max_power'],
        "EV Spawn Multiplier": ['spawn_multiplier'],
        "EV Charge Efficiency": ['ev', 'charge_efficiency'],
        "EV Discharge Efficiency": ['ev', 'discharge_efficiency'],
        "EV Desired Capacity (%)": ['ev', 'desired_capacity'],
        "Discharge Price Factor": ['discharge_price_factor'],
        "Inflexible Loads Forecast Mean": ['inflexible_loads', 'forecast_mean'],
        "Solar Power Forecast Mean": ['solar_power', 'forecast_mean'],
    }

    # --- 2. User Configuration ---
    print("--- Sensitivity Analysis Configuration ---")

    # Select base scenario
    config_path = "ev2gym/example_config_files/"
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    base_scenario_path = select_from_list([os.path.basename(s) for s in available_scenarios], "Select the BASE scenario file to modify:")
    base_scenario_full_path = os.path.join(config_path, base_scenario_path)

    # Select parameter to analyze
    param_name = select_from_list(list(key_parameters.keys()), "Select the parameter for sensitivity analysis:")
    param_path = key_parameters[param_name]

    # Define range for the selected parameter
    print(f"\nDefine the range for '{param_name}':")
    start_val = float(get_user_input("Enter the START value", 10))
    end_val = float(get_user_input("Enter the END value", 100))
    steps = int(get_user_input("Enter the NUMBER of steps", 5))
    param_range = np.linspace(start_val, end_val, steps)

    print(f"\nAnalysis will run for '{param_name}' with values: {param_range}")

    # --- 3. Setup Simulation Parameters ---
    # These are set to reasonable defaults but could be made interactive
    is_thesis_mode = True
    MAX_CS = calculate_max_cs(config_path)
    algorithms_to_run = get_algorithms(MAX_CS, is_thesis_mode)
    reward_func = reward_module.FastProfitAdaptiveReward # Default reward function
    num_simulations = 1 # Keep this low for faster analysis
    model_dir = './saved_models/multi_scenario_20240901/' # Assuming a default pre-trained model
    is_multi_scenario = True
    price_data_file = './ev2gym/data/Netherlands_day-ahead-2015-2024.csv'

    # --- 4. Run Analysis ---
    base_results_path = f'./results/sensitivity_{param_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(base_results_path, exist_ok=True)
    
    all_results_summary = []

    for i, value in enumerate(param_range):
        print(f"\n{'='*80}")
        print(f"--- Running Step {i+1}/{steps}: {param_name} = {value} ---")
        print(f"{'='*80}")

        # Load and modify config
        with open(base_scenario_full_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set the parameter value, ensuring correct type (int or float)
        final_value = int(value) if isinstance(param_range, np.ndarray) and np.issubdtype(param_range.dtype, np.integer) else float(value)
        set_nested_dict_value(config, param_path, final_value)

        # Create a single temporary config file in the root directory
        temp_config_path = 'temp_sensitivity_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Run the benchmark in silent mode
        aggregated_stats = run_benchmark(
            config_files=[temp_config_path],
            reward_func=reward_func,
            algorithms_to_run=algorithms_to_run,
            num_simulations=num_simulations,
            model_dir=model_dir,
            is_multi_scenario=is_multi_scenario,
            price_data_file=price_data_file,
            generate_plots=False
        )

        # --- 5. Collect Results ---
        for algo_name, stats in aggregated_stats.items():
            # This part needs to be adapted based on the structure of aggregated_stats
            # Assuming aggregated_stats is {algo: {'mean': {metric: val}, 'std': {metric: val}}}
            row = {
                'Algorithm': algo_name,
                'parameter_name': param_name,
                'parameter_value': value
            }
            for metric, mean_val in stats.get('mean', {}).items():
                std_val = stats.get('std', {}).get(metric, 0)
                # This format is for plotting later, we store raw numbers first
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
            all_results_summary.append(row)

    # --- 6. Aggregate and Plot Results ---
    # Clean up the temporary config file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

    if not all_results_summary:
        print("\nNo results were generated. Exiting analysis.")
        return

    final_df = pd.DataFrame(all_results_summary)
    final_df.to_csv(os.path.join(base_results_path, "sensitivity_analysis_summary.csv"), index=False)
    
    print(f"\n{'='*80}")
    print(f"--- Sensitivity Analysis Complete ---")
    print(f"Full summary saved to: {os.path.join(base_results_path, 'sensitivity_analysis_summary.csv')}")
    print(f"{'='*80}")

    # Plotting
    # Identify metrics to plot (those with a _mean suffix)
    metrics_to_plot = [col.replace('_mean', '') for col in final_df.columns if col.endswith('_mean')]
    algorithms = final_df['Algorithm'].unique()

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        for algo in algorithms:
            algo_df = final_df[final_df['Algorithm'] == algo]
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if mean_col in algo_df.columns:
                plt.plot(algo_df['parameter_value'], algo_df[mean_col], marker='o', linestyle='-', label=algo)
                if std_col in algo_df.columns:
                    plt.fill_between(algo_df['parameter_value'], 
                                     algo_df[mean_col] - algo_df[std_col], 
                                     algo_df[mean_col] + algo_df[std_col], 
                                     alpha=0.2)

        plt.title(f"Sensitivity of '{metric}' to '{param_name}'")
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.grid(True, which='both', linestyle='--')
        plt.legend()
        
        plot_filename = os.path.join(base_results_path, f"sensitivity_{param_name.replace(' ', '_')}_vs_{metric.replace(' ', '_')}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved to: {plot_filename}")

if __name__ == "__main__":
    run_sensitivity_analysis()
