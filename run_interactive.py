import os
import inspect
from glob import glob
import subprocess
import time
import yaml
import json
from collections import defaultdict

# RL library imports
from stable_baselines3 import SAC, DDPG, TD3, PPO
from sb3_contrib import TQC

# Imports from the custom ev2gym library and other scripts
from run_experiments import (
    calculate_max_cs,
    get_algorithms,
    train_rl_models_if_requested,
    run_benchmark
)
from ev2gym.rl_agent import reward as reward_module

# --- User Interface Utility Functions ---

def get_interactive_input(prompt, default=None):
    """Helper function to get user input with a default value."""
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input or default

def select_from_list(items, prompt, multiple=False, default_choice=1):
    """Helper function to let the user select one or more items from a list."""
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        display_name = item if isinstance(item, str) else item[0]
        print(f"  {i+1}. {os.path.basename(display_name).replace('.yaml', '')}")
    
    if not items:
        print("No items available for selection.")
        return [] if multiple else None

    if multiple:
        choices_str = get_interactive_input(f"Select one or more (e.g., '1 3', 'all')", 'all').lower()
        if 'all' in choices_str:
            return items
        try:
            indices = [int(i) - 1 for i in choices_str.split()]
            return [items[i] for i in indices if 0 <= i < len(items)]
        except (ValueError, IndexError):
            print("Invalid selection. Using all items.")
            return items
    else:
        try:
            choice_str = get_interactive_input(f"Choice", str(default_choice))
            choice = int(choice_str) - 1
            if 0 <= choice < len(items):
                return items[choice]
            else:
                raise IndexError
        except (ValueError, IndexError):
            print(f"Invalid selection. Using the default choice ({default_choice}).")
            return items[default_choice - 1]

def get_available_rl_algorithms():
    """
    Returns a dictionary of RL algorithms compatible with continuous action spaces.
    DQN and other discrete action algorithms are excluded.
    """
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

def select_model_directory(prompt="Select the set of models to load:"):
    """Lets the user select a directory from ./saved_models/."""
    saved_models_dir = './saved_models/'
    if not os.path.exists(saved_models_dir) or not os.listdir(saved_models_dir):
        print(f"\nERROR: No folders found in '{saved_models_dir}'. Please run training first.")
        return None, False

    available_models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
    selected_model_name = select_from_list(available_models, prompt, multiple=False)
    
    if not selected_model_name:
        return None, False

    model_dir = os.path.join(saved_models_dir, selected_model_name)
    
    # Multi-scenario mode is now the standard
    is_multi_scenario = True
    print(f"\nModels selected from: {model_dir} (multi-scenario mode by default)")
    return model_dir, is_multi_scenario

# --- Main Flows: Training and Plotting ---

def run_training_flow():
    """Manages the workflow for training new models."""
    print("\n--- Starting Training Flow ---")

    # 1. Select algorithms to train
    available_rl_algos = get_available_rl_algorithms()
    selected_algo_names = select_from_list(
        list(available_rl_algos.keys()), 
        "Select the RL algorithms to train:", 
        multiple=True
    )
    if not selected_algo_names:
        print("No algorithms selected. Training cancelled.")
        return

    algorithms_to_train = {k: available_rl_algos[k] for k in selected_algo_names}
    print(f"Algorithms to be trained: {list(algorithms_to_train.keys())}")

    # 2. Training configuration (simplified)
    config_path = "ev2gym/example_config_files/"
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    
    print("\nTraining will be performed in 'Dynamic Parameters' mode.")
    print("This mode trains models with dynamically varied parameters for greater robustness.")
    base_scenario_path = select_from_list(available_scenarios, "Select a BASE scenario for parameter randomization:")
    
    steps_for_training = int(get_interactive_input("For how many total training steps?", "100000"))
    session_name = get_interactive_input("Enter a name for this training session", f"dynamic_{'_'.join(selected_algo_names).lower()}_{time.strftime('%Y%m%d')}")
    model_dir = f'./saved_models/{"" .join(c for c in session_name if c.isalnum() or c in ("_", "-")).rstrip()}/'
    os.makedirs(model_dir, exist_ok=True)

    # 3. Select reward and price file
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    selected_reward_tuple = select_from_list(available_rewards, "Choose the reward function:", default_choice=1)
    selected_reward_func = selected_reward_tuple[1]

    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    default_price_index = available_price_files.index(default_price_file) + 1 if default_price_file in available_price_files else 1
    selected_price_file_name = select_from_list(available_price_files, "Select the CSV file for energy prices:", default_choice=default_price_index)
    selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)

    # 4. Execute training
    print(f"\n--- Starting training in folder: {model_dir} ---")
    train_rl_models_if_requested(
        scenarios_to_test=[base_scenario_path],
        selected_reward_func=selected_reward_func,
        algorithms_to_run=algorithms_to_train,
        is_multi_scenario=True, # Always True with the new mode
        model_dir=model_dir,
        selected_price_file_abs_path=selected_price_file_abs_path,
        steps_for_training=steps_for_training,
        training_mode='dynamic', # Robust default mode
        session_name=session_name
    )
    print("\n--- Training Completed ---")

def run_plotting_flow():
    """Manages the workflow for benchmarking and plotting existing models."""
    print("\n--- Starting Plotting Flow ---")

    # 1. Select the models folder
    model_dir, is_multi_scenario = select_model_directory()
    if not model_dir:
        return

    # 2. Detect available algorithms in that folder
    available_model_files = glob(os.path.join(model_dir, '*_model.zip'))
    trained_rl_algos = [os.path.basename(f).replace('_model.zip', '').replace('_', '+').upper() for f in available_model_files]
    
    if not trained_rl_algos:
        print(f"No .zip model files found in {model_dir}. Cannot proceed.")
        return
        
    print(f"Trained RL algorithms found in this session: {trained_rl_algos}")

    # 3. Get base algorithms (Heuristics, MPC) and RL
    MAX_CS = calculate_max_cs("ev2gym/example_config_files/")
    all_base_algos = get_algorithms(MAX_CS, is_thesis_mode=True)
    baselines = {k: v for k, v in all_base_algos.items() if v[1] is None}
    
    # Use the new dynamic function to get RL algorithm definitions
    all_rl_definitions = get_available_rl_algorithms()
    available_rl_from_files = {k: v for k, v in all_rl_definitions.items() if k in trained_rl_algos}

    # Merge the definitions of all available algorithms for this plot
    all_available_definitions = {**baselines, **available_rl_from_files}

    # 4. Select algorithms to plot
    # The order here determines how they appear in the selection list
    plot_candidates = sorted(list(baselines.keys())) + sorted(list(available_rl_from_files.keys()))
    selected_for_plot = select_from_list(plot_candidates, "Select the algorithms to compare in the benchmark:", multiple=True)
    
    if not selected_for_plot:
        print("No algorithms selected for plotting. Cancelled.")
        return

    algorithms_to_run = {k: all_available_definitions[k] for k in selected_for_plot}
    print(f"\nAlgorithms that will be run in the benchmark: {list(algorithms_to_run.keys())}")

    # 5. Benchmark configuration
    config_path = "ev2gym/example_config_files/"
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    benchmark_scenarios = select_from_list(available_scenarios, "Select the scenarios for the BENCHMARK:", multiple=True)
    
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    selected_reward_tuple = select_from_list(available_rewards, "Choose the reward function (make sure it's the same as for training):", default_choice=1)
    selected_reward_func = selected_reward_tuple[1]

    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    default_price_index = available_price_files.index(default_price_file) + 1 if default_price_file in available_price_files else 1
    selected_price_file_name = select_from_list(available_price_files, "Select the CSV file for energy prices:", default_choice=default_price_index)
    selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)

    num_sims = int(get_interactive_input("How many evaluation simulations per scenario?", "1"))

    # 6. Execute benchmark
    print("\n--- Starting Benchmark and Plot Generation ---")
    run_benchmark(
        config_files=benchmark_scenarios,
        reward_func=selected_reward_func,
        algorithms_to_run=algorithms_to_run,
        num_simulations=num_sims,
        model_dir=model_dir,
        is_multi_scenario=is_multi_scenario,
        price_data_file=selected_price_file_abs_path
    )
    print("\n--- EXECUTION COMPLETED ---")


def main():
    """Main function to orchestrate the execution."""
    
    # --- Preliminary execution of Fit_battery.py ---
    if get_interactive_input("Do you want to run 'Fit_battery.py' to calibrate the degradation model? (y/n)", "n").lower() == 'y':
        print("--- Running Fit_battery.py ---")
        try:
            subprocess.run(["python", "Fit_battery.py"], check=True)
            print("--- Fit_battery.py completed. ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERROR: {e}. The script will proceed with existing parameters.")

    # --- Main Menu ---
    while True:
        choice = get_interactive_input("\nWhat do you want to do?\n  1. Train new RL models\n  2. Run benchmark and plot results of existing models\n\nChoice", "2")
        if choice == '1':
            run_training_flow()
        elif choice == '2':
            run_plotting_flow()
        else:
            print("Invalid choice.")

        if get_interactive_input("\nDo you want to perform another operation? (y/n)", "n").lower() != 'y':
            break
            
    print("\n--- Program terminated. ---")

if __name__ == "__main__":
    main()
