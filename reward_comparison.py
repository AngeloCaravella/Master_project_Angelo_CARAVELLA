import os
import shutil
import datetime
from run_experiments import train_rl_models_if_requested, run_benchmark, get_algorithms, plot_performance_metrics
from ev2gym.rl_agent import reward as reward_module
from stable_baselines3 import SAC

def run_reward_comparison():
    """
    Trains SAC models with different reward functions and benchmarks them
    to compare their performance on key metrics, including training time.
    """
    print("--- Starting Reward Function Comparison Analysis ---")

    # --- Configuration ---
    REWARD_FUNCTIONS_TO_TEST = {
        "FastProfitAdaptive": reward_module.FastProfitAdaptiveReward,
        "ProfitMax_Penalty": reward_module.ProfitMax_TrPenalty_UserIncentives,
    }
    
    ALGORITHM_TO_TRAIN = {
        "SAC": (None, SAC, {})
    }

    TRAINING_SCENARIO = ["ev2gym/example_config_files/Realistic.yaml"]
    BENCHMARK_SCENARIOS = ["ev2gym/example_config_files/Realistic.yaml", "ev2gym/example_config_files/V2GProfitMax.yaml"]
    TRAINING_STEPS = 20000

    # --- Main Logic ---
    
    overall_results = {}
    training_times_by_reward = {}
    base_model_dir = f'./temp_reward_comparison_models_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'

    # 1. Training Loop
    print(f"\n--- Phase 1: Training {len(REWARD_FUNCTIONS_TO_TEST)} agents ---")
    for reward_name, reward_func in REWARD_FUNCTIONS_TO_TEST.items():
        print(f"\n-- Training agent with reward function: {reward_name} --")
        
        model_dir = os.path.join(base_model_dir, reward_name)
        os.makedirs(model_dir, exist_ok=True)
        
        session_name = f"reward_comp_{reward_name}"

        training_times = train_rl_models_if_requested(
            scenarios_to_test=TRAINING_SCENARIO,
            selected_reward_func=reward_func,
            algorithms_to_run=ALGORITHM_TO_TRAIN,
            is_multi_scenario=False,
            model_dir=model_dir,
            selected_price_file_abs_path=None,
            steps_for_training=TRAINING_STEPS,
            training_mode='single',
            session_name=session_name
        )
        
        if "SAC" in training_times:
            training_times_by_reward[reward_name] = training_times["SAC"]
        
        print(f"-- Finished training for {reward_name}. Model saved in {model_dir} --")

    # 2. Benchmarking Loop
    print(f"\n--- Phase 2: Benchmarking trained agents ---")
    for reward_name, reward_func in REWARD_FUNCTIONS_TO_TEST.items():
        print(f"\n-- Benchmarking agent trained with: {reward_name} --")
        
        model_dir = os.path.join(base_model_dir, reward_name)

        benchmark_results = run_benchmark(
            config_files=BENCHMARK_SCENARIOS,
            reward_func=lambda env, *args: 0,
            algorithms_to_run=ALGORITHM_TO_TRAIN,
            num_simulations=3,
            model_dir=model_dir,
            is_multi_scenario=False,
            generate_plots=False
        )
        
        if "SAC" in benchmark_results:
            overall_results[reward_name] = benchmark_results["SAC"]
            if reward_name in training_times_by_reward:
                overall_results[reward_name]['mean']['training_time'] = training_times_by_reward[reward_name]
                overall_results[reward_name]['std']['training_time'] = 0 # Std is 0 for a single training run
        
        print(f"-- Finished benchmarking for {reward_name} --")

    # 3. Plotting Results
    print("\n--- Phase 3: Generating Comparison Plot ---")
    if not overall_results:
        print("No results were generated from the benchmark. Cannot create comparison plot.")
        return

    save_path = f'./results/reward_comparison_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(save_path, exist_ok=True)
    
    plot_performance_metrics(
        stats_collection=overall_results,
        save_path=save_path,
        scenario_name="RewardFunctionComparison",
        algorithms_to_plot=list(REWARD_FUNCTIONS_TO_TEST.keys()),
        total_evs=25, 
        num_charging_points=20,
        transformer_limit=100
    )
    
    print(f"\n--- Reward Comparison Analysis Complete ---")
    print(f"Comparison plot saved in: {save_path}")

    # 4. Cleanup
    try:
        shutil.rmtree(base_model_dir)
        print(f"Cleaned up temporary model directory: {base_model_dir}")
    except OSError as e:
        print(f"Error cleaning up temporary directory {base_model_dir}: {e}")


if __name__ == "__main__":
    run_reward_comparison()
