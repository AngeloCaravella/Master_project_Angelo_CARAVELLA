import os
import yaml
import numpy as np
import pandas as pd
import time
import gymnasium as gym
from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import ast

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.pulp_mpc_sensitivity import OnlineMPC_Solver
from ev2gym.rl_agent.state import V2G_profit_max_loads
from ev2gym.rl_agent import reward as reward_module

# Helper function to parse "value ± std" strings
def parse_mean_std_string(s):
    """Parses a string like '17.00 ± 2.16' into a tuple (mean, std)."""
    if pd.isna(s) or not isinstance(s, str):
        return (np.nan, np.nan)
    
    s = s.strip()
    parts = s.split(' ± ')
    if len(parts) == 2:
        try:
            mean_val = float(parts[0])
            std_val = float(parts[1])
            return (mean_val, std_val)
        except ValueError:
            return (np.nan, np.nan)
    elif len(parts) == 1:  # Case where there's no std dev (e.g., "17.00")
        try:
            mean_val = float(parts[0])
            return (mean_val, 0.0)  # Assume 0 std dev if not present
        except ValueError:
            return (np.nan, np.nan)
    return (np.nan, np.nan)  # Default for unparseable strings


def plot_mpc_results(all_results_df, output_dir, plot_filename_prefix):
    """Generates plots for MPC sensitivity analysis results."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Performance Metrics ---
    metrics_to_plot = {
        'total_profits': 'Total Profits (€)',
        'average_user_satisfaction': 'Average User Satisfaction (%)',
        'transformer_overload_kwh': 'Transformer Overload (kWh)'
    }
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    fig.suptitle('MPC Performance Metrics by Configuration', fontsize=16)

    for i, (metric_base_name, title) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        metric_mean_col = f"{metric_base_name}_mean"
        metric_std_col = f"{metric_base_name}_std"
        if metric_mean_col in all_results_df.columns:
            ax.bar(
                all_results_df['Configuration'],
                all_results_df[metric_mean_col],
                yerr=all_results_df[metric_std_col],
                capsize=5,
                color='skyblue'
            )
            ax.set_title(title)
            ax.set_ylabel(title.split('(')[-1].replace(')', ''))
            ax.tick_params(axis='x', rotation=90)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax.set_title(f"{title} (Data Missing - Column '{metric_mean_col}' not found)")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}_performance.png"))
    plt.close(fig)
    print(f"Plot saved: {plot_filename_prefix}_performance.png")

    # --- Plot 2: Solver Time Metrics ---
    time_metrics_to_plot = {
        'average_solver_time_per_step': 'Avg Solver Time per Step (s)',
        'total_solver_time': 'Total Solver Time (s)'
    }

    fig, axes = plt.subplots(1, len(time_metrics_to_plot), figsize=(5 * len(time_metrics_to_plot), 6))
    if len(time_metrics_to_plot) == 1:
        axes = [axes]
    fig.suptitle('MPC Solver Time Metrics by Configuration', fontsize=16)

    for i, (metric_base_name, title) in enumerate(time_metrics_to_plot.items()):
        ax = axes[i]
        metric_mean_col = f"{metric_base_name}_mean"
        metric_std_col = f"{metric_base_name}_std"
        if metric_mean_col in all_results_df.columns:
            ax.bar(
                all_results_df['Configuration'],
                all_results_df[metric_mean_col],
                yerr=all_results_df[metric_std_col],
                capsize=5,
                color='lightcoral'
            )
            ax.set_title(title)
            ax.set_ylabel('Time (s)')
            ax.tick_params(axis='x', rotation=90)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax.set_title(f"{title} (Data Missing - Column '{metric_mean_col}' not found)")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}_solver_time.png"))
    plt.close(fig)
    print(f"Plot saved: {plot_filename_prefix}_solver_time.png")

    # --- Plot 3: Solver Status Counts ---
    status_metrics_to_plot = {
        'solver_timeouts': 'Solver Timeouts',
        'infeasible_steps': 'Infeasible Steps',
        'non_optimal_steps': 'Non-Optimal Steps (excluding Timeout/Infeasible)'
    }

    fig, axes = plt.subplots(1, len(status_metrics_to_plot), figsize=(5 * len(status_metrics_to_plot), 6))
    if len(status_metrics_to_plot) == 1:
        axes = [axes]
    fig.suptitle('MPC Solver Status Counts by Configuration', fontsize=16)

    for i, (metric_base_name, title) in enumerate(status_metrics_to_plot.items()):
        ax = axes[i]
        metric_mean_col = f"{metric_base_name}_mean"
        metric_std_col = f"{metric_base_name}_std"
        if metric_mean_col in all_results_df.columns:
            ax.bar(
                all_results_df['Configuration'],
                all_results_df[metric_mean_col],
                yerr=all_results_df[metric_std_col],
                capsize=5,
                color='lightgreen'
            )
            ax.set_title(title)
            ax.set_ylabel('Number of Steps')
            ax.tick_params(axis='x', rotation=90)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax.set_title(f"{title} (Data Missing - Column '{metric_mean_col}' not found)")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}_solver_status.png"))
    plt.close(fig)
    print(f"Plot saved: {plot_filename_prefix}_solver_status.png")

    # --- Plot 4: Adaptive Horizon History ---
    adaptive_configs = [cfg for cfg in all_results_df['Configuration'] if 'Adaptive' in cfg]

    if adaptive_configs and 'adaptive_horizon_history' in all_results_df.columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle('Adaptive Horizon (H) Over Time', fontsize=16)
        
        plot_data = all_results_df[all_results_df['Configuration'].isin(adaptive_configs)][
            ['Configuration', 'adaptive_horizon_history']
        ]
        
        colors = plt.colormaps.get_cmap('tab10')
        found_adaptive_data_to_plot = False

        for i, (idx, row) in enumerate(plot_data.iterrows()):
            config_name = row['Configuration']
            history = row['adaptive_horizon_history']

            # --- FIX robust per NaN, None o liste vuote ---
            if history is None or (isinstance(history, float) and np.isnan(history)):
                continue
            if isinstance(history, (list, np.ndarray)) and len(history) == 0:
                continue
            # ------------------------------------------------

            if not all(isinstance(h, list) for h in history):
                history = [history]

            non_empty_histories = [h for h in history if h]
            if not non_empty_histories:
                continue

            max_len = max(len(h) for h in non_empty_histories)
            if max_len == 0:
                continue

            padded_histories = []
            for h in non_empty_histories:
                padded_histories.append(h + [h[-1]] * (max_len - len(h)))

            avg_history = np.mean(padded_histories, axis=0)
            std_history = np.std(padded_histories, axis=0)

            if len(avg_history) > 0:
                color = colors(i / len(plot_data))
                ax.plot(range(len(avg_history)), avg_history, label=config_name, color=color, linewidth=2)
                ax.fill_between(
                    range(len(avg_history)),
                    avg_history - std_history,
                    avg_history + std_history,
                    color=color,
                    alpha=0.1
                )
                found_adaptive_data_to_plot = True

        if found_adaptive_data_to_plot:
            ax.set_xlabel('Simulation Step')
            ax.set_ylabel('Prediction Horizon (H)')
            ax.set_title('Average Adaptive Prediction Horizon Over Simulation Steps')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title='Configuration')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}_adaptive_horizon.png"))
            plt.close(fig)
            print(f"Plot saved: {plot_filename_prefix}_adaptive_horizon.png")
        else:
            print(f"No valid adaptive horizon data found to plot. Skipping {plot_filename_prefix}_adaptive_horizon.png")
            plt.close(fig)
    else:
        print("No adaptive configurations or 'adaptive_horizon_history' column found. Skipping adaptive horizon plot.")


def run_mpc_sensitivity_analysis(
    base_config_file: str,
    reward_func,
    price_data_file: str,
    mpc_configs: list,
    num_simulations: int = 1,
    solver_timeout: int = 50,
    output_dir: str = "./mpc_sensitivity_results"
):
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = defaultdict(lambda: defaultdict(list))
    all_adaptive_horizon_histories = defaultdict(list)

    print(f"\n--- Starting MPC Sensitivity Analysis for {os.path.basename(base_config_file)} ---")
    print(f"Testing {len(mpc_configs)} configurations over {num_simulations} simulations each.")

    for config_idx, mpc_params in enumerate(mpc_configs):
        np_val = mpc_params.get('prediction_horizon', 'Adaptive')
        nc_val = mpc_params.get('control_horizon', 'Adaptive')
        use_adaptive = mpc_params.get('use_adaptive_horizon', False)
        
        config_name = f"MPC_Np{np_val}_Nc{nc_val}"
        if use_adaptive:
            config_name = f"MPC_Adaptive_hmax{mpc_params.get('h_max')}_Nc{nc_val}"
        
        print(f"\n--- Running configuration: {config_name} ({config_idx + 1}/{len(mpc_configs)}) ---")

        for sim_num in range(num_simulations):
            print(f"  Simulation {sim_num + 1}/{num_simulations}...")

            env = EV2Gym(
                config_file=base_config_file,
                generate_rnd_game=True,
                reward_function=reward_func,
                state_function=V2G_profit_max_loads,
                price_data_file=price_data_file,
                record_historic_soc=True
            )

            mpc_solver = OnlineMPC_Solver(
                env=env,
                prediction_horizon=mpc_params.get('prediction_horizon', 10),
                control_horizon=mpc_params.get('control_horizon', 'half'),
                use_adaptive_horizon=mpc_params.get('use_adaptive_horizon', False),
                h_max=mpc_params.get('h_max', 5),
                h_min=mpc_params.get('h_min', 2),
                lyapunov_alpha=mpc_params.get('lyapunov_alpha', 0.1),
                solver_timeout=solver_timeout
            )

            obs, _ = env.reset()
            done = False
            total_solver_time = 0
            step_count = 0

            solver_timeout_count = 0
            infeasible_count = 0
            non_optimal_count = 0

            current_sim_adaptive_horizon_history = []

            while not done:
                step_count += 1
                solver_start_time = time.time()
                action, solver_status = mpc_solver.get_action(env)
                solver_end_time = time.time()

                solver_duration = solver_end_time - solver_start_time
                total_solver_time += solver_duration

                if solver_status == 'Timeout':
                    solver_timeout_count += 1
                elif solver_status == 'Infeasible':
                    infeasible_count += 1
                elif solver_status != 'Optimal':
                    non_optimal_count += 1

                if use_adaptive:
                    current_sim_adaptive_horizon_history.append(mpc_solver.current_H)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            unwrapped_env = env
            ev_load = np.sum(unwrapped_env.cs_power, axis=0)
            inflexible_load = np.sum(unwrapped_env.tr_inflexible_loads, axis=0)
            solar_power = np.sum(unwrapped_env.tr_solar_power, axis=0)
            limit = unwrapped_env.transformers[0].max_power
            timescale = unwrapped_env.timescale

            if isinstance(limit, (list, np.ndarray)):
                limit = limit[0]

            net_load = ev_load + inflexible_load - solar_power
            overload_power = np.maximum(0, net_load - limit)
            overload_kwh = np.sum(overload_power * (timescale / 60.0))

            stats = unwrapped_env.stats
            stats['transformer_overload_kwh'] = overload_kwh
            stats['total_solver_time'] = total_solver_time
            stats['average_solver_time_per_step'] = total_solver_time / step_count if step_count > 0 else 0
            stats['solver_timeouts'] = solver_timeout_count
            stats['infeasible_steps'] = infeasible_count
            stats['non_optimal_steps'] = non_optimal_count

            for metric, value in stats.items():
                all_results[config_name][metric].append(value)

            if use_adaptive:
                all_adaptive_horizon_histories[config_name].append(current_sim_adaptive_horizon_history)

            env.close()

    summary_data_for_df = []
    summary_data_for_display = []

    for config_name, metrics in all_results.items():
        row_df = {'Configuration': config_name}
        row_display = {'Configuration': config_name}

        for metric, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)

            row_df[f"{metric}_mean"] = mean_val
            row_df[f"{metric}_std"] = std_val
            row_display[metric] = f"{mean_val:.2f} ± {std_val:.2f}" if len(values) > 1 else f"{mean_val:.2f}"

        if config_name in all_adaptive_horizon_histories:
            row_df['adaptive_horizon_history'] = all_adaptive_horizon_histories[config_name]
        else:
            row_df['adaptive_horizon_history'] = np.nan

        summary_data_for_df.append(row_df)
        summary_data_for_display.append(row_display)

    if summary_data_for_df:
        df_summary_numeric = pd.DataFrame(summary_data_for_df)
        df_summary_display = pd.DataFrame(summary_data_for_display)

        csv_path = os.path.join(
            output_dir, f"mpc_sensitivity_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_summary_display.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n--- Sensitivity Analysis Complete ---")
        print(f"Results saved to: {csv_path}")
        print("\nAggregated Results:")
        print(df_summary_display.to_string())

        plot_filename_prefix = f"mpc_sensitivity_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        plot_mpc_results(df_summary_numeric, output_dir, plot_filename_prefix)

    else:
        print("\n--- No results to display. ---")


def get_interactive_input(prompt, default=None):
    """Helper function to get user input with a default value."""
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input or default


if __name__ == "__main__":
    choice = get_interactive_input(
        "\nWhat do you want to do?\n  1. Run new MPC sensitivity analysis\n  2. Load results from existing CSV file and plot\nChoice", "1"
    )

    if choice == '1':
        BASE_CONFIG_FILE = "C:\\Users\\angel\\OneDrive\\Desktop\\Project_Master\\ev2gym\\example_config_files\\Critic.yaml"
        PRICE_DATA_FILE = "C:\\Users\\angel\\OneDrive\\Desktop\\Project_Master\\ev2gym\\data\\Netherlands_day-ahead-2015-2024.csv"
        
        SELECTED_REWARD_FUNCTION = reward_module.FastProfitAdaptiveReward

        MPC_CONFIGURATIONS = [
            {'prediction_horizon': 5, 'control_horizon': 1, 'use_adaptive_horizon': False},
            {'prediction_horizon': 15, 'control_horizon': 5, 'use_adaptive_horizon': False},
            {'prediction_horizon': 5, 'control_horizon': 5, 'use_adaptive_horizon': False},
            {'prediction_horizon': 10, 'control_horizon': 10, 'use_adaptive_horizon': False},
            {'prediction_horizon': 25, 'control_horizon': 10, 'use_adaptive_horizon': False},
            
            {'use_adaptive_horizon': True, 'h_max': 5, 'control_horizon': 1},
            {'use_adaptive_horizon': True, 'h_max': 15, 'control_horizon': 5},
            {'use_adaptive_horizon': True, 'h_max': 5, 'control_horizon': 5},
            {'use_adaptive_horizon': True, 'h_max': 10, 'control_horizon': 10},
            {'use_adaptive_horizon': True, 'h_max': 25, 'control_horizon': 10},
        ]

        NUM_SIMULATIONS_PER_CONFIG = 3
        SOLVER_TIMEOUT_SECONDS = 50

        run_mpc_sensitivity_analysis(
            base_config_file=BASE_CONFIG_FILE,
            reward_func=SELECTED_REWARD_FUNCTION,
            price_data_file=PRICE_DATA_FILE,
            mpc_configs=MPC_CONFIGURATIONS,
            num_simulations=NUM_SIMULATIONS_PER_CONFIG,
            solver_timeout=SOLVER_TIMEOUT_SECONDS
        )
    elif choice == '2':
        print("\n--- Load results from existing CSV file ---")
        results_dir = "./mpc_sensitivity_results/"
        available_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")), reverse=True)
        
        if not available_files:
            print(f"No .csv files found in {results_dir}. Please run a new analysis first or place a results file there.")
        else:
            print("\nAvailable result files:")
            for i, f in enumerate(available_files):
                print(f"  {i+1}. {os.path.basename(f)}")
            
            file_choice = get_interactive_input("Select a file to load", "1")
            try:
                selected_file_idx = int(file_choice) - 1
                if 0 <= selected_file_idx < len(available_files):
                    file_path = available_files[selected_file_idx]
                    loaded_df_display = pd.read_csv(file_path, encoding='utf-8')
                    
                    # Create a new DataFrame to hold numeric mean/std values for plotting
                    loaded_df_numeric = pd.DataFrame({'Configuration': loaded_df_display['Configuration']})

                    # List of columns that contain "value ± std" strings
                    # Exclude 'Configuration' and 'adaptive_horizon_history'
                    cols_to_parse = [col for col in loaded_df_display.columns if col not in ['Configuration', 'adaptive_horizon_history']]

                    for col in cols_to_parse:
                        if col in loaded_df_display.columns:
                            parsed_data = loaded_df_display[col].apply(parse_mean_std_string)
                            loaded_df_numeric[f"{col}_mean"] = [x[0] for x in parsed_data]
                            loaded_df_numeric[f"{col}_std"] = [x[1] for x in parsed_data]
                        else:
                            print(f"Warning: Column '{col}' not found in loaded CSV. Skipping parsing for this column.")

                    # Handle adaptive_horizon_history separately
                    if 'adaptive_horizon_history' in loaded_df_display.columns:
                        loaded_df_numeric['adaptive_horizon_history'] = loaded_df_display['adaptive_horizon_history'].apply(
                            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else x
                        )
                    else:
                        loaded_df_numeric['adaptive_horizon_history'] = np.nan

                    print(f"Successfully loaded and parsed data from {file_path}")
                    print("\nLoaded Numeric Data Head:\n", loaded_df_numeric.head())
                    
                    output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
                    plot_filename_prefix = f"mpc_sensitivity_loaded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    plot_mpc_results(loaded_df_numeric, output_dir, plot_filename_prefix)
                    print("\nPlots generated from loaded data.")
                else:
                    print("Invalid file selection.")
            except ValueError as e:
                print(f"Invalid input or error during file processing: {e}. Please enter a number.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    else:
        print("Invalid choice. Exiting.")
