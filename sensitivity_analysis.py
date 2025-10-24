import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
import seaborn as sns
import copy
import time

# Import SALib
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer

# Import from project
from run_experiments import run_benchmark, get_algorithms, calculate_max_cs
from ev2gym.rl_agent import reward as reward_module

# Import RL libraries
from stable_baselines3 import SAC, DDPG, TD3, PPO
from sb3_contrib import TQC

# --- UI Helper Functions ---

def get_user_input(prompt, default=None):
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input or default

def select_from_list(items, prompt, multiple=False, default_choice=1):
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"  {i+1}. {os.path.basename(item)}")
    
    if not items:
        print("No items available.")
        return None if not multiple else []

    if multiple:
        choices_str = get_user_input(f"Select one or more (e.g., '1, 3', 'all')", 'all').lower()
        if 'all' in choices_str:
            return items
        try:
            cleaned_str = choices_str.replace(',', ' ')
            indices = [int(i.strip()) - 1 for i in cleaned_str.split() if i.strip().isdigit()]
            if not indices:
                raise ValueError("No valid indices found.")
            return [items[i] for i in indices if 0 <= i < len(items)]
        except (ValueError, IndexError):
            print("Invalid selection. Returning all items.")
            return items
    else: # single selection
        try:
            choice_str = get_user_input(f"Choice", str(default_choice))
            choice = int(choice_str) - 1
            return items[choice] if 0 <= choice < len(items) else items[default_choice - 1]
        except (ValueError, IndexError):
            return items[default_choice - 1]

# --- Core Logic Functions ---

def create_comparison_plot(results_dir):
    """
    Loads Morris results from a directory, normalizes them, and creates a
    comparative grouped bar chart for a predefined set of algorithms.
    """
    print("\n--- Generating Normalized Morris Comparison Plot ---")

    # As per user request, define one of each type for a focused comparison
    target_algorithms = {
        'heuristic': 'ALAP',
        'mpc': 'MPC_2',
        'rl': 'SAC'
    }

    all_results = {}
    
    print(f"Loading results from: {results_dir}")

    for algo_type, algo_name in target_algorithms.items():
        file_path_pattern = os.path.join(results_dir, f"morris_indices_{algo_name}.csv")
        found_files = glob(file_path_pattern)
        
        if not found_files:
            print(f"WARNING: Could not find results file for '{algo_name}'. Skipping for comparison plot.")
            continue
        
        print(f"-> Found results for '{algo_name}'.")
        df = pd.read_csv(found_files[0])
        
        # Normalize mu_star by dividing by the sum
        df['mu_star_norm'] = df['mu_star'] / df['mu_star'].sum()
        all_results[algo_name] = df.set_index('names')

    if len(all_results) < 2:
        print("Comparison plot requires at least two algorithm results. Aborting plot generation.")
        return

    # --- Plotting ---
    plot_data = []
    for algo_name, df in all_results.items():
        df_reset = df.reset_index()
        df_reset['Algorithm'] = algo_name
        plot_data.append(df_reset)
    
    combined_df = pd.concat(plot_data, ignore_index=True)
    param_order = combined_df['names'].unique()

    fig, ax = plt.subplots(figsize=(18, 10))
    
    n_algos = len(all_results)
    n_params = len(param_order)
    bar_width = 0.8 / n_algos
    index = np.arange(n_params)
    colors = plt.cm.get_cmap('viridis', n_algos)

    for i, algo_name in enumerate(all_results.keys()):
        algo_df = combined_df[combined_df['Algorithm'] == algo_name].set_index('names').loc[param_order]
        bar_positions = index - (bar_width * (n_algos - 1) / 2) + (i * bar_width)
        ax.bar(bar_positions, algo_df['mu_star_norm'], width=bar_width, label=algo_name, color=colors(i))

    ax.set_xlabel('Parameters', fontsize=14)
    ax.set_ylabel('Normalized μ* (Relative Importance)', fontsize=14)
    ax.set_title('Comparison of Normalized Morris Sensitivity (μ*) Across Key Algorithms', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(param_order, rotation=45, ha="right")
    ax.legend(title="Algorithm")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "morris_comparison_plot.png")
    plt.savefig(save_path)
    print(f"\nComparison plot saved to: {save_path}")
    plt.close(fig)

def get_available_rl_algorithms():
    """Returns a dictionary of RL algorithms compatible with continuous action spaces."""
    return {
        "SAC": (None, SAC, {}),
        "DDPG": (None, DDPG, {}),
        "TD3": (None, TD3, {}),
        "TQC": (None, TQC, {
            'policy_kwargs': dict(n_quantiles=25, n_critics=2),
            'top_quantiles_to_drop_per_net': 5
        }),
        "PPO": (None, PPO, {}),
    }

def select_and_validate_rl_models():
    """Handles the selection of a model directory and the RL algorithms to be analyzed."""
    saved_models_dir = './saved_models/'
    if not os.path.exists(saved_models_dir) or not os.listdir(saved_models_dir):
        print(f"ERROR: No directory found in '{saved_models_dir}'. Please train models first.")
        sys.exit(1)

    available_model_dirs = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
    selected_model_dir_name = select_from_list(available_model_dirs, "Select the training session to analyze:")
    model_dir = os.path.join(saved_models_dir, selected_model_dir_name)
    is_multi_scenario = True

    available_model_files = glob(os.path.join(model_dir, '*_model.zip'))
    if not available_model_files:
        print(f"ERROR: No model files (*_model.zip) found in {model_dir}.")
        sys.exit(1)
    
    trained_rl_algos_in_dir = sorted([os.path.basename(f).replace('_model.zip', '').replace('_', '+').upper() for f in available_model_files])
    print(f"\nRL algorithms found in this session: {trained_rl_algos_in_dir}")

    selected_rl_names = select_from_list(trained_rl_algos_in_dir, "Select RL algorithms to analyze:", multiple=True)
    if not selected_rl_names:
        print("No algorithm selected. Analysis cancelled.")
        sys.exit(0)

    all_rl_definitions = get_available_rl_algorithms()
    algorithms_to_run = {}
    for name in selected_rl_names:
        if name in all_rl_definitions:
            algorithms_to_run[name] = all_rl_definitions[name]
        else:
            print(f"WARNING: Definition for algorithm '{name}' not found and will be skipped.")

    include_baselines = get_user_input("\nInclude heuristics and MPC in comparison plots? (y/n)", "y").lower() == 'y'
    if include_baselines:
        print("Heuristics and MPC will be added to the plots.")
        all_base_algos = get_algorithms(calculate_max_cs("ev2gym/example_config_files/"), is_thesis_mode=True)
        baselines = {k: v for k, v in all_base_algos.items() if v[1] is None}
        algorithms_to_run.update(baselines)
    
    print(f"\nFinal algorithms for benchmark: {list(algorithms_to_run.keys())}")
    return model_dir, is_multi_scenario, algorithms_to_run

def run_morris_analysis(base_scenario_full_path, key_parameters, PREDEFINED_LEVELS):
    """Guides user through Morris analysis setup and execution (sequential execution)."""
    print("\n--- Sensitivity Analysis: Morris Method (Sequential Execution) ---")

    selected_param_names = select_from_list(list(key_parameters.keys()), "Select parameters to analyze:", multiple=True)
    
    if not isinstance(selected_param_names, list):
        selected_param_names = [selected_param_names]

    bounds = [[min(PREDEFINED_LEVELS[p]), max(PREDEFINED_LEVELS[p])] for p in selected_param_names]
    problem = {'num_vars': len(selected_param_names), 'names': selected_param_names, 'bounds': bounds}

    num_trajectories = int(get_user_input("Number of trajectories (N)", 10))
    # Automatically determine the number of levels (p) for the Morris sampler grid
    # based on the maximum number of levels in the selected parameters.
    num_levels = max(len(PREDEFINED_LEVELS[p]) for p in selected_param_names)
    
    # Ensure num_levels is an even number, as required by the Morris sampler.
    if num_levels % 2 != 0:
        num_levels += 1
        
    print(f"Info: Using num_levels={num_levels} for the Morris sampler grid (derived from selected parameters).")
    param_values = morris_sampler.sample(problem, N=num_trajectories, num_levels=num_levels)

    # --- MODIFIED ALGORITHM SELECTION ---
    print("\n--- Algorithm Selection for Morris Analysis ---")
    all_algos = get_algorithms(calculate_max_cs("ev2gym/example_config_files/"), is_thesis_mode=True)
    selected_algo_names = select_from_list(list(all_algos.keys()), "Select algorithms to run Morris Analysis on:", multiple=True)
    
    if not selected_algo_names:
        print("No algorithms selected. Cancelling.")
        return

    algorithms_to_analyze = {k: all_algos[k] for k in selected_algo_names}

    model_dir, is_multi_scenario = None, False
    if any(v[1] is not None for v in algorithms_to_analyze.values()):
        saved_models_dir = './saved_models/'
        available_model_dirs = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
        selected_model_dir_name = select_from_list(available_model_dirs, "Select the training session for the RL models:")
        model_dir = os.path.join(saved_models_dir, selected_model_dir_name)
        is_multi_scenario = True
    # --- END MODIFICATION ---

    reward_func = reward_module.FastProfitAdaptiveReward
    price_data_file = './ev2gym/data/Netherlands_day-ahead-2015-2024.csv'
    output_metric = 'total_profits'
    base_results_path = f'./sensitivity_analysis_results/Morris_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(base_results_path, exist_ok=True)

    with open(base_scenario_full_path, 'r') as f:
        base_config = yaml.safe_load(f)

    for algo_name, algo_details in algorithms_to_analyze.items():
        print(f"\n{'='*80}")
        print(f"--- Starting Morris Analysis for: {algo_name} ---")
        
        Y = np.zeros(param_values.shape[0])

        for i, X in enumerate(param_values):
            print(f"  Running sample {i+1}/{len(param_values)}...")
            config = copy.deepcopy(base_config)

            for j, param_name_iter in enumerate(selected_param_names):
                sampled_value = X[j]
                
                # Snap to the nearest predefined level for discrete parameters
                levels = PREDEFINED_LEVELS[param_name_iter]
                final_value = levels[np.abs(np.array(levels) - sampled_value).argmin()]
                
                set_nested_dict_value(config, key_parameters[param_name_iter], final_value)

            temp_config_path = f'temp_morris_config_{algo_name}.yaml'
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)

            aggregated_stats = run_benchmark(
                config_files=[temp_config_path],
                reward_func=reward_func,
                algorithms_to_run={algo_name: algo_details},
                num_simulations=1,
                model_dir=model_dir,
                is_multi_scenario=is_multi_scenario,
                price_data_file=price_data_file,
                generate_plots=False
            )
            
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

            Y[i] = aggregated_stats.get(algo_name, {}).get('mean', {}).get(output_metric, 0)
            print(f"  -> Result: {Y[i]:.2f}")

        Si = morris_analyzer.analyze(problem, param_values, Y, conf_level=0.95)
        results_df = pd.DataFrame(Si)
        results_df.to_csv(os.path.join(base_results_path, f"morris_indices_{algo_name}.csv"), index=False)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(Si['mu_star'], Si['sigma'], s=100)
        for i, txt in enumerate(Si['names']): 
            ax.annotate(txt, (Si['mu_star'][i], Si['sigma'][i]), xytext=(5,5), textcoords='offset points')
        ax.set_title(f"Morris Indices for {algo_name} (Metric: {output_metric})")
        ax.set_xlabel("μ* (Total Influence)"); ax.set_ylabel("σ (Interactions & Non-linearities)")
        ax.grid(True)
        plt.savefig(os.path.join(base_results_path, f"morris_plot_{algo_name}.png")); plt.close()
        print(f"Results for {algo_name} saved in: {base_results_path}")

    print(f"\n--- Morris Analysis Complete. ---")
    if get_user_input("\nGenerate normalized comparison plot for key algorithms (ALAP, MPC_2, SAC)? (y/n)", "y").lower() == 'y':
        create_comparison_plot(base_results_path)

def run_oat_analysis(config_path, key_parameters, PREDEFINED_LEVELS, base_scenario_full_path):
    """Runs the One-at-a-Time sensitivity analysis for one or more parameters."""
    print("\n--- Sensitivity Analysis: One-at-a-Time ---")
    
    all_possible_params = list(PREDEFINED_LEVELS.keys())
    selected_params = select_from_list(all_possible_params, "Select parameter(s) to analyze:", multiple=True)

    if not selected_params:
        print("No parameters selected. Aborting OAT analysis.")
        return

    model_dir, is_multi_scenario, algorithms_to_run = select_and_validate_rl_models()
    num_simulations = int(get_user_input("Number of simulations per point", "1"))
    reward_func = reward_module.FastProfitAdaptiveReward
    price_data_file = './ev2gym/data/Netherlands_day-ahead-2015-2024.csv'
    
    # Load base config once to get fixed parameter values for the legend
    with open(base_scenario_full_path, 'r') as f:
        base_config = yaml.safe_load(f)

    for param_name in selected_params:
        print(f"\n\n{'='*80}\nRunning OAT Analysis for: {param_name}\n{'='*80}")
        param_path = key_parameters[param_name]
        param_range = PREDEFINED_LEVELS[param_name]
        
        base_results_path = f'./sensitivity_analysis_results/OAT_{param_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(base_results_path, exist_ok=True)

        all_results_summary = []
        for i, value in enumerate(param_range):
            print(f"--- Executing step {i+1}/{len(param_range)}: {param_name} = {value} ---")
            config = copy.deepcopy(base_config)
            
            final_value = int(value) if isinstance(value, int) else float(value)
            set_nested_dict_value(config, param_path, final_value)

            temp_config_path = 'temp_sensitivity_config.yaml'
            with open(temp_config_path, 'w') as f: yaml.dump(config, f)

            aggregated_stats = run_benchmark(
                config_files=[temp_config_path],
                reward_func=reward_func,
                algorithms_to_run=algorithms_to_run,
                num_simulations=num_simulations,
                model_dir=model_dir,
                is_multi_scenario=is_multi_scenario,
                price_data_file=price_data_file,
                generate_plots=False,
                seed=42
            )

            for algo_name, stats in aggregated_stats.items():
                row = {'Algorithm': algo_name, 'parameter_name': param_name, 'parameter_value': value}
                row.update({f"{metric}_mean": mean_val for metric, mean_val in stats.get('mean', {}).items()})
                all_results_summary.append(row)

        if os.path.exists(temp_config_path): os.remove(temp_config_path)
        if not all_results_summary: continue

        final_df = pd.DataFrame(all_results_summary)
        final_df.to_csv(os.path.join(base_results_path, "sensitivity_summary.csv"), index=False)
        print(f"\n--- Analysis for {param_name} Complete. Results saved in: {base_results_path} ---")

        # --- Plotting Logic ---
        fixed_params_text_list = [f"Base Scenario: {os.path.basename(base_scenario_full_path)}", "Fixed Parameters:"]
        for i, (p_name, p_path) in enumerate(key_parameters.items()):
            if p_name != param_name:
                try:
                    val = base_config
                    for key in p_path:
                        val = val[key]
                    fixed_params_text_list.append(f"{i+1}: {p_name} = {val}")
                except KeyError:
                    fixed_params_text_list.append(f"{i+1}: {p_name} = N/A")
        fixed_params_text = " | ".join(fixed_params_text_list)

        metrics_to_plot = [col.replace('_mean', '') for col in final_df.columns if col.endswith('_mean')]
        for metric in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(data=final_df, x='parameter_value', y=f'{metric}_mean', hue='Algorithm', marker='o', style='Algorithm', ax=ax)
            
            ax.set_title(f"Sensitivity of '{metric}' to '{param_name}'")
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.grid(True, which='both', linestyle='--')
            ax.legend(title='Algorithm')
            
            fig.text(0.5, 0.01, fixed_params_text, wrap=True, ha='center', va='bottom', fontsize=8)
            fig.tight_layout(rect=[0, 0.05, 1, 1])

            plot_filename = os.path.join(base_results_path, f"sensitivity_{param_name.replace(' ', '_')}_vs_{metric}.png")
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"Plot saved to: {plot_filename}")

def run_scenario_comparison_analysis(config_path):
    """Runs a comparison across different scenario files."""
    print("\n--- Scenario Comparison Analysis ---")

    model_dir, is_multi_scenario, algorithms_to_run = select_and_validate_rl_models()

    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    selected_scenario_paths = select_from_list(available_scenarios, "Select scenarios to compare:", multiple=True)

    num_simulations = int(get_user_input("Number of simulations per scenario", "1"))
    reward_func = reward_module.FastProfitAdaptiveReward
    price_data_file = './ev2gym/data/Netherlands_day-ahead-2015-2024.csv'
    base_results_path = f'./sensitivity_analysis_results/ScenarioComp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(base_results_path, exist_ok=True)

    all_results_summary = []
    for scenario_path in selected_scenario_paths:
        scenario_name = os.path.basename(scenario_path).replace('.yaml', '')
        print(f"\n--- Running scenario: {scenario_name} ---")
        aggregated_stats = run_benchmark(
            config_files=[scenario_path],
            reward_func=reward_func,
            algorithms_to_run=algorithms_to_run,
            num_simulations=num_simulations,
            model_dir=model_dir,
            is_multi_scenario=is_multi_scenario,
            price_data_file=price_data_file,
            generate_plots=False,
            seed=42
        )
        for algo_name, stats in aggregated_stats.items():
            row = {'Algorithm': algo_name, 'scenario_name': scenario_name}
            row.update({f"{metric}_mean": mean_val for metric, mean_val in stats.get('mean', {}).items()})
            all_results_summary.append(row)

    if not all_results_summary: return

    final_df = pd.DataFrame(all_results_summary)
    final_df.to_csv(os.path.join(base_results_path, "scenario_comparison_summary.csv"), index=False)
    print(f"\n--- Analysis Complete. Results saved in: {base_results_path} ---")

    metrics_to_plot = [col.replace('_mean', '') for col in final_df.columns if col.endswith('_mean')]
    for metric in metrics_to_plot:
        try:
            plt.figure(figsize=(14, 7))
            sns.barplot(data=final_df, x='scenario_name', y=f'{metric}_mean', hue='Algorithm')
            plt.title(f"Comparison of '{metric}' across Scenarios")
            plt.xlabel('Scenario')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Algorithm')
            plt.tight_layout()
            save_path = os.path.join(base_results_path, f"comparison_{metric}.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Could not generate plot for metric '{metric}'. Error: {e}")

# --- Main Function ---

def set_nested_dict_value(d, path, value):
    for key in path[:-1]: d = d.setdefault(key, {})
    d[path[-1]] = value

def main():
    """Main function to orchestrate the sensitivity analysis."""
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
    analysis_methods = ['One-at-a-Time (OAT)', 'Morris Method', 'Scenario Comparison']
    selected_analysis = select_from_list(analysis_methods, "Select analysis method:")

    if selected_analysis == 'Scenario Comparison':
        run_scenario_comparison_analysis(config_path)
    else:
        available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
        base_scenario_path = select_from_list(available_scenarios, "Select the BASE scenario to modify:")
        if selected_analysis == 'Morris Method':
            run_morris_analysis(base_scenario_path, key_parameters, PREDEFINED_LEVELS)
        else:
            run_oat_analysis(config_path, key_parameters, PREDEFINED_LEVELS, base_scenario_path)

if __name__ == "__main__":
    main()
