
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
import itertools

# Import SALib for Morris method
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer

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

def select_multiple_from_list(items, prompt):
    """Displays a list of items and asks the user to select one or more."""
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"  {i+1}. {item}")
    print(f"  {len(items)+1}. ALL")
    choices_str = get_user_input(f"Enter the numbers of your choices (comma-separated), or '{len(items)+1}' for ALL", str(len(items)+1)) 
    
    if choices_str == str(len(items)+1):
        return items

    try:
        choices_idx = [int(c.strip()) - 1 for c in choices_str.split(',')]
        selected_items = [items[i] for i in choices_idx if 0 <= i < len(items)]
        if not selected_items:
            raise ValueError("Empty selection.")
        return selected_items
    except (ValueError, IndexError):
        print("Invalid selection. Defaulting to all algorithms.")
        return items

def select_model_directory():
    saved_models_dir = './saved_models/'
    if not os.path.exists(saved_models_dir) or not os.listdir(saved_models_dir):
        print("\nERRORE: Nessun modello addestrato trovato in './saved_models/'. Esegui prima l'addestramento.")
        return None, False # Return None for model_dir and False for is_multi_scenario

    available_models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
    selected_model_name = select_from_list(available_models, "Seleziona il set di modelli da caricare per l'analisi di sensitività:")
    model_dir = os.path.join(saved_models_dir, selected_model_name)
    
    is_multi_scenario = False
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    if os.path.exists(metadata_path):
        is_multi_scenario = True
        print(f"Rilevato file di metadati: i modelli sono multi-scenario.")
    else:
        is_multi_scenario = any(keyword in selected_model_name.lower() for keyword in ['multi', 'curriculum', 'shuffled', 'random'])
        print(f"Nessun file di metadati. Modalità rilevata dal nome: {'multi-scenario' if is_multi_scenario else 'scenario singolo'}")
    
    print(f"\nModelli selezionati da: {model_dir}")
    return model_dir, is_multi_scenario

# --- Analysis-specific Functions ---

def _setup_common_simulation_parameters(config_path):
    """Handles the common setup steps for any analysis type."""
    master_seed = 42
    print(f"\n--- Using Master Seed: {master_seed} for simulations to ensure EV population is fixed (if num_simulations=1). ---")
    
    is_thesis_mode = True
    MAX_CS = calculate_max_cs(config_path)
    all_available_algorithms = get_algorithms(MAX_CS, is_thesis_mode)

    selected_algo_names = select_multiple_from_list(
        list(all_available_algorithms.keys()), 
        "Select the algorithm(s) for the analysis:"
    )
    algorithms_to_run = {name: all_available_algorithms[name] for name in selected_algo_names}
    print(f"\nWill run analysis for the following algorithms: {list(algorithms_to_run.keys())}")

    reward_func = reward_module.FastProfitAdaptiveReward
    num_simulations = int(get_user_input("Enter number of simulations per run", 1))
    
    model_dir, is_multi_scenario = select_model_directory()
    
    price_data_file = './ev2gym/data/Netherlands_day-ahead-2015-2024.csv'
    
    return (algorithms_to_run, reward_func, model_dir, is_multi_scenario, 
            price_data_file, num_simulations, master_seed)

def run_scenario_comparison_analysis(config_path, key_parameters):
    """Runs a comparison across different scenario files."""
    print("\n--- Scenario Comparison Analysis ---")

    # 1. Setup common parameters
    (algorithms_to_run, reward_func, model_dir, is_multi_scenario, 
     price_data_file, num_simulations, master_seed) = _setup_common_simulation_parameters(config_path)

    if model_dir is None:
        return

    # 2. Select scenarios to compare
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    selected_scenarios_names = select_multiple_from_list(
        [os.path.basename(s) for s in available_scenarios],
        "Select the scenarios to compare:"
    )
    selected_scenario_paths = [os.path.join(config_path, s) for s in selected_scenarios_names]

    # 3. Run Analysis
    base_results_path = f'C:/Users/angel/OneDrive/Desktop/Project_Master/sensitivity_analysis_results/scenario_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(base_results_path, exist_ok=True)
    
    all_results_summary = []

    for scenario_path in selected_scenario_paths:
        scenario_name = os.path.basename(scenario_path).replace('.yaml', '')
        print(f"\n{'='*80}")
        print(f"--- Running Scenario: {scenario_name} ---")
        print(f"{'='*80}")

        aggregated_stats = run_benchmark(
            config_files=[scenario_path],
            reward_func=reward_func,
            algorithms_to_run=algorithms_to_run,
            num_simulations=num_simulations,
            model_dir=model_dir,
            is_multi_scenario=is_multi_scenario,
            price_data_file=price_data_file,
            generate_plots=False,
            seed=master_seed 
        )

        # 4. Collect Results
        for algo_name, stats in aggregated_stats.items():
            row = {
                'Algorithm': algo_name,
                'scenario_name': scenario_name
            }
            for metric, mean_val in stats.get('mean', {}).items():
                std_val = stats.get('std', {}).get(metric, 0)
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
            all_results_summary.append(row)

    # 5. Aggregate and Plot Results
    if not all_results_summary:
        print("\nNo results were generated. Exiting analysis.")
        return

    final_df = pd.DataFrame(all_results_summary)
    summary_csv_path = os.path.join(base_results_path, "scenario_comparison_summary.csv")
    final_df.to_csv(summary_csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"--- Scenario Comparison Complete ---")
    print(f"Full summary saved to: {summary_csv_path}")
    print(f"{'='*80}")

    metrics_to_plot = [col.replace('_mean', '') for col in final_df.columns if col.endswith('_mean')]

    for metric in metrics_to_plot:
        mean_col = f"{metric}_mean"
        if mean_col not in final_df.columns:
            continue

        try:
            pivot_df = final_df.pivot(index='scenario_name', columns='Algorithm', values=mean_col)
            
            ax = pivot_df.plot(kind='bar', figsize=(14, 7), width=0.8, 
                               title=f"Comparison of '{metric}' across Scenarios",
                               grid=True)
            
            ax.set_ylabel(metric)
            ax.set_xlabel("Scenario")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Algorithm')
            plt.tight_layout()
            
            plot_filename = os.path.join(base_results_path, f"comparison_vs_{metric}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Could not generate plot for metric '{metric}'. Error: {e}")

def run_oat_analysis(config_path, key_parameters, PREDEFINED_LEVELS, base_scenario_full_path):
    """Runs the One-at-a-Time sensitivity analysis."""
    print("\n--- One-at-a-Time Sensitivity Analysis ---")
    
    param_name = select_from_list(
        list(PREDEFINED_LEVELS.keys()), 
        "Select the parameter for sensitivity analysis:"
    )
    param_path = key_parameters[param_name]
    param_range = PREDEFINED_LEVELS[param_name]
    steps = len(param_range)

    print(f"\nAnalysis will run for '{param_name}' with predefined levels.")
    print(f"Levels: {param_range}")

    # Setup common parameters
    (algorithms_to_run, reward_func, model_dir, is_multi_scenario, 
     price_data_file, num_simulations, master_seed) = _setup_common_simulation_parameters(config_path)

    if model_dir is None:
        return

    # Run Analysis
    base_results_path = f'C:/Users/angel/OneDrive/Desktop/Project_Master/sensitivity_analysis_results/sensitivity_{param_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(base_results_path, exist_ok=True)
    
    all_results_summary = []

    for i, value in enumerate(param_range):
        print(f"\n{'='*80}")
        print(f"--- Running Step {i+1}/{steps}: {param_name} = {value:.4f} ---")
        print(f"{'='*80}")

        with open(base_scenario_full_path, 'r') as f:
            config = yaml.safe_load(f)
        
        final_value = int(value) if np.issubdtype(type(value), np.integer) else float(value)
        set_nested_dict_value(config, param_path, final_value)

        # --- DEBUG PRINT: Show key parameters for this run ---
        print("\n--- DEBUG: Current Parameter Values for this Run ---")
        for p_name, p_path in key_parameters.items():
            try:
                current_val = config
                for key in p_path:
                    current_val = current_val[key]
                print(f"  - {p_name}: {current_val}")
            except KeyError:
                print(f"  - {p_name}: Not Found in Config")
        print("--------------------------------------------------\n")
        # --- END DEBUG PRINT ---

        temp_config_path = 'temp_sensitivity_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        aggregated_stats = run_benchmark(
            config_files=[temp_config_path],
            reward_func=reward_func,
            algorithms_to_run=algorithms_to_run,
            num_simulations=num_simulations,
            model_dir=model_dir,
            is_multi_scenario=is_multi_scenario,
            price_data_file=price_data_file,
            generate_plots=False,
            seed=master_seed
        )

        # Collect Results
        for algo_name, stats in aggregated_stats.items():
            row = {
                'Algorithm': algo_name,
                'parameter_name': param_name,
                'parameter_value': value
            }
            for metric, mean_val in stats.get('mean', {}).items():
                std_val = stats.get('std', {}).get(metric, 0)
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val
            all_results_summary.append(row)

    # Aggregate and Plot Results
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

    metrics_to_plot = [col.replace('_mean', '') for col in final_df.columns if col.endswith('_mean')]
    algorithms = final_df['Algorithm'].unique()

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        for algo in algorithms:
            algo_df = final_df[final_df['Algorithm'] == algo]
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if mean_col in algo_df.columns:
                sorted_df = algo_df.sort_values(by='parameter_value')
                plt.plot(sorted_df['parameter_value'], sorted_df[mean_col], marker='o', linestyle='-', label=algo)
                if std_col in sorted_df.columns and num_simulations > 1:
                    plt.fill_between(sorted_df['parameter_value'], 
                                     sorted_df[mean_col] - sorted_df[std_col], 
                                     sorted_df[mean_col] + sorted_df[std_col], 
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

    # Special plot for Discharge Price Factor
    if param_name == "Discharge Price Factor":
        print(f"\nGenerating special plot for '{param_name}'...")
        try:
            price_df = pd.read_csv(price_data_file)
            original_prices = price_df['Price (EUR/MWhe)'].head(48)
            hours = np.arange(len(original_prices))

            plt.figure(figsize=(15, 8))
            plt.plot(hours, original_prices, label='Charge Price (Original / Factor = 1.0)', color='blue', linestyle='--', linewidth=2.5)

            for factor in param_range:
                if factor == 1.0:
                    continue
                discharge_prices = original_prices * factor
                plt.plot(hours, discharge_prices, label=f'Discharge Price (Factor: {factor:.2f})', marker='.', linestyle='-')

            plt.title("Effect of 'Discharge Price Factor' on Price Curves (First 48 Hours)")
            plt.xlabel("Hour")
            plt.ylabel("Price (EUR/MWh)")
            plt.legend()
            plt.grid(True, which='both', linestyle='--')

            plot_filename = os.path.join(base_results_path, "price_curves_vs_discharge_factor.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Price curve comparison plot saved to: {plot_filename}")

        except FileNotFoundError:
            print(f"\nWARNING: Price data file not found at '{price_data_file}'. Skipping price curve plot.")
        except KeyError:
            print(f"\nWARNING: Could not find 'Price (EUR/MWhe)' column in '{price_data_file}'. Skipping price curve plot.")

def run_morris_analysis(base_scenario_full_path, key_parameters, config_path, PREDEFINED_LEVELS):
    # This function remains largely unchanged but could be refactored to use 
    # _setup_common_simulation_parameters if needed in the future.
    """Guides user through Morris analysis setup and execution for each selected algorithm."""
    print("\n--- Morris Sensitivity Analysis Configuration ---")

    # ... (rest of the original morris analysis code)
    # Note: This part is kept as is, since it has a more complex setup loop
    # that doesn't fit the new common setup function directly.
    selected_param_names = select_multiple_from_list(
        list(key_parameters.keys()),
        "Select parameters for Morris analysis:"
    )

    bounds = []
    for param_name in selected_param_names:
        if param_name in PREDEFINED_LEVELS:
            min_val = min(PREDEFINED_LEVELS[param_name])
            max_val = max(PREDEFINED_LEVELS[param_name])
            bounds.append([min_val, max_val])
            print(f"  Bounds for '{param_name}' set to [{min_val}, {max_val}] from PREDEFINED_LEVELS.")
        else:
            print(f"\nDefine bounds for '{param_name}':")
            min_val = float(get_user_input(f"Enter MIN value", 0))
            max_val = float(get_user_input(f"Enter MAX value", 1))
            bounds.append([min_val, max_val])

    num_trajectories = int(get_user_input("\nEnter the number of trajectories (e.g., 10)", 10))
    num_levels = int(get_user_input("Enter the number of levels (e.g., 4 or 6, preferably an even number)", 4))

    problem = {
        'num_vars': len(selected_param_names),
        'names': selected_param_names,
        'bounds': bounds
    }

    print(f"Generating {num_trajectories} trajectories for Morris analysis...")
    param_values = morris_sampler.sample(problem, N=num_trajectories, num_levels=num_levels)

    is_thesis_mode = True
    MAX_CS = calculate_max_cs(config_path)
    all_available_algorithms = get_algorithms(MAX_CS, is_thesis_mode)
    selected_algo_names = select_multiple_from_list(
        list(all_available_algorithms.keys()),
        "Select the algorithm(s) for the analysis:"
    )
    algorithms_to_run = {name: all_available_algorithms[name] for name in selected_algo_names}
    
    reward_func = reward_module.FastProfitAdaptiveReward
    num_simulations = 1
    
    model_dir, is_multi_scenario = select_model_directory()
    if model_dir is None:
        return

    price_data_file = './ev2gym/data/Netherlands_day-ahead-2015-2024.csv'
    output_metric = 'total_profits'

    base_results_path = f'C:/Users/angel/OneDrive/Desktop/Project_Master/sensitivity_analysis_results/morris_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(base_results_path, exist_ok=True)
    print(f"\nSaving all results for this run in: {base_results_path}")

    for algo_name, algo_details in algorithms_to_run.items():
        print(f"\n{'='*80}")
        print(f"--- Starting Morris Analysis for Algorithm: {algo_name} ---")
        print(f"{'='*80}")

        algo_results_path = os.path.join(base_results_path, algo_name)
        os.makedirs(algo_results_path, exist_ok=True)

        Y = np.zeros([param_values.shape[0]])

        for i, X in enumerate(param_values):
            print(f"\n--- Running Sample {i+1}/{param_values.shape[0]} for {algo_name} ---")

            with open(base_scenario_full_path, 'r') as f:
                config = yaml.safe_load(f)

            for j, param_name_iter in enumerate(selected_param_names):
                param_path = key_parameters[param_name_iter]
                value = X[j]

                if param_name_iter in ["Number of Charging Stations", "Transformer Max Power (kW)", "EV Spawn Multiplier", "Inflexible Loads Forecast Mean", "Solar Power Forecast Mean"]:
                    final_value = int(round(value))
                else:
                    final_value = float(value)
                
                set_nested_dict_value(config, param_path, final_value)
                print(f"  {param_name_iter}: {final_value:.4f}")

            sanitized_algo_name = algo_name.replace('+', '_').replace('.', '_')
            temp_config_path = f'temp_morris_config_{sanitized_algo_name}.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)

            aggregated_stats = run_benchmark(
                config_files=[temp_config_path],
                reward_func=reward_func,
                algorithms_to_run={{algo_name: algo_details}},
                num_simulations=num_simulations,
                model_dir=model_dir,
                is_multi_scenario=is_multi_scenario,
                price_data_file=price_data_file,
                generate_plots=False
            )
            
            Y[i] = aggregated_stats[algo_name]['mean'].get(output_metric, 0)
            print(f"  Result ({output_metric}): {Y[i]:.2f}")

        Si = morris_analyzer.analyze(problem, param_values, Y, conf_level=0.95)
        
        print(f"\n--- Morris Analysis Results for {algo_name} ---")
        print(Si)
        
        results_df = Si.to_df()
        csv_path = os.path.join(algo_results_path, f"morris_analysis_results_{algo_name}.csv")
        results_df.to_csv(csv_path)
        print(f"Morris analysis results for {algo_name} saved to {csv_path}")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(Si['mu_star'], Si['sigma'], c=Si['mu_star'], cmap='viridis', s=120, alpha=0.8)
        
        texts = []
        for i, txt in enumerate(Si['names']):
            texts.append(ax.text(Si['mu_star'][i], Si['sigma'][i], txt, fontsize=9))
        
        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
        except ImportError:
            print("Consider installing `adjustText` for better label placement: `pip install adjustText`")
            for t in texts:
                t.set_visible(False)
            for i, txt in enumerate(Si['names']):
                ax.annotate(txt, (Si['mu_star'][i], Si['sigma'][i]), xytext=(5,5), textcoords='offset points', fontsize=9)

        mean_mu_star = np.mean(Si['mu_star'])
        mean_sigma = np.mean(Si['sigma'])
        ax.axvline(mean_mu_star, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(mean_sigma, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(f"Morris Method Results for {algo_name}", fontsize=16)
        ax.set_xlabel("μ* (Mean of absolute elementary effects)", fontsize=12)
        ax.set_ylabel("σ (Standard deviation of elementary effects)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()

        plot_filename = os.path.join(algo_results_path, f"morris_analysis_plot_{algo_name}.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"Morris analysis plot for {algo_name} saved to: {plot_filename}")

        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    print(f"\n{'='*80}")
    print("--- All Morris Analyses Complete ---")
    print(f"{'='*80}")

def run_sensitivity_analysis():
    """Main function to configure and run the sensitivity analysis."""
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

    PREDEFINED_LEVELS = {
        "Number of Charging Stations": [5, 15, 25, 50, 100],
        "Transformer Max Power (kW)": [25, 50, 100, 200, 400],
        "EV Spawn Multiplier": [1, 3, 5, 7, 10],
        "EV Charge Efficiency": [0.80, 0.90, 0.95, 0.99],
        "EV Discharge Efficiency": [0.70, 0.85, 0.90, 0.98],
        "EV Desired Capacity (%)": [0.6, 0.75, 0.85, 0.95, 1.0],
        "Discharge Price Factor": [0.5, 0.8, 1.0, 1.5, 2.0],
        "Inflexible Loads Forecast Mean": [0, 15, 30, 45, 60],
        "Solar Power Forecast Mean": [0, 20, 40, 60, 80],
    }

    config_path = "ev2gym/example_config_files/"
    
    print("--- Sensitivity Analysis Configuration ---")
    analysis_methods = ['One-at-a-Time (Predefined Levels)', 'Morris', 'Scenario Comparison']
    selected_analysis = select_from_list(analysis_methods, "Select the analysis method:")

    if selected_analysis == 'Scenario Comparison':
        run_scenario_comparison_analysis(config_path, key_parameters)
        return

    # Common for OAT and Morris
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    base_scenario_path = select_from_list([os.path.basename(s) for s in available_scenarios], "Select the BASE scenario file to modify:")
    base_scenario_full_path = os.path.join(config_path, base_scenario_path)

    if selected_analysis == 'Morris':
        run_morris_analysis(base_scenario_full_path, key_parameters, config_path, PREDEFINED_LEVELS)
    else: # Default to OAT
        run_oat_analysis(config_path, key_parameters, PREDEFINED_LEVELS, base_scenario_full_path)

if __name__ == "__main__":
    run_sensitivity_analysis()
