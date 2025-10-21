import os
import shutil
import datetime
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from run_experiments import train_rl_models_if_requested, run_benchmark, get_algorithms, plot_performance_metrics, ProgressCallback, TrainingPlotCallback
from ev2gym.rl_agent import reward as reward_module
from ev2gym.rl_agent.state import V2G_profit_max_loads
from stable_baselines3 import DDPG
import time
import torch
from glob import glob # Added for dynamic scenario loading

def run_reward_comparison():
    """
    Trains DDPG models with different reward functions and benchmarks them
    to compare their performance on key metrics, including training time.
    """
    print("--- Starting Reward Function Comparison Analysis ---")

    # --- Configuration ---
    REWARD_FUNCTIONS_TO_TEST = {
        "FastProfitAdaptive": reward_module.FastProfitAdaptiveReward,
        "ProfitMax_Penalty": reward_module.ProfitMax_TrPenalty_UserIncentives,
        "CurriculumLearning": None, # Placeholder for curriculum learning, handled specially
    }
    
    ALGORITHM_TO_TRAIN = {
        "DDPG": (None, DDPG, {})
    }

    config_files_path = "ev2gym/example_config_files/"
    all_yaml_files = sorted(glob(os.path.join(config_files_path, "*.yaml")))
    TRAINING_SCENARIO = all_yaml_files
    BENCHMARK_SCENARIOS = all_yaml_files
    TRAINING_STEPS = 10000

    # --- Main Logic ---
    
    overall_results = {}
    training_times_by_reward = {}
    base_model_dir = f'./temp_reward_comparison_models_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'

    # 1. Training Loop
    print(f"\n--- Phase 1: Training {len(REWARD_FUNCTIONS_TO_TEST)} agents ---")
    for reward_name, reward_func in REWARD_FUNCTIONS_TO_TEST.items():
        if reward_name == "CurriculumLearning":
            print(f"\n-- Training agent with Curriculum Learning (ProfitMax then FastProfitAdaptive) --")
            cl_model_dir = os.path.join(base_model_dir, reward_name)
            os.makedirs(cl_model_dir, exist_ok=True)
            
            # Define training steps for each stage
            training_steps_stage1 = TRAINING_STEPS // 2
            training_steps_stage2 = TRAINING_STEPS - training_steps_stage1 # Ensure total steps match
            
            # Get DDPG class and kwargs
            _, rl_class, rl_kwargs = ALGORITHM_TO_TRAIN["DDPG"]

            # --- Stage 1: Train with ProfitMax_TrPenalty_UserIncentives ---
            print(f"---- Stage 1: Training with ProfitMax_TrPenalty_UserIncentives for {training_steps_stage1} steps ----")
            train_env_stage1 = Monitor(gym.make('ev2gym.models.ev2gym_env:EV2Gym', config_file=TRAINING_SCENARIO[0], generate_rnd_game=True, reward_function=reward_module.ProfitMax_TrPenalty_UserIncentives, state_function=V2G_profit_max_loads, price_data_file=None))
            
            model_stage1 = rl_class("MlpPolicy", train_env_stage1, verbose=0, device="cuda" if torch.cuda.is_available() else "cpu", **rl_kwargs)
            
            start_time_stage1 = time.process_time()
            model_stage1.learn(total_timesteps=training_steps_stage1, callback=[ProgressCallback(training_steps_stage1), TrainingPlotCallback("DDPG_CL_Stage1", "reward_comp_CL_stage1")])
            end_time_stage1 = time.process_time()
            
            stage1_model_path = os.path.join(cl_model_dir, "ddpg_stage1_model.zip")
            model_stage1.save(stage1_model_path)
            train_env_stage1.close()

            # --- Stage 2: Continue training with FastProfitAdaptiveReward ---
            print(f"---- Stage 2: Continuing training with FastProfitAdaptiveReward for {training_steps_stage2} steps ----")
            train_env_stage2 = Monitor(gym.make('ev2gym.models.ev2gym_env:EV2Gym', config_file=TRAINING_SCENARIO[0], generate_rnd_game=True, reward_function=reward_module.FastProfitAdaptiveReward, state_function=V2G_profit_max_loads, price_data_file=None))
            
            # Load the partially trained model
            model_stage2 = rl_class.load(stage1_model_path, env=train_env_stage2, device="cuda" if torch.cuda.is_available() else "cpu")
            
            start_time_stage2 = time.process_time()
            model_stage2.learn(total_timesteps=training_steps_stage2, callback=[ProgressCallback(training_steps_stage2), TrainingPlotCallback("DDPG_CL_Stage2", "reward_comp_CL_stage2")])
            end_time_stage2 = time.process_time()
            
            final_cl_model_path = os.path.join(cl_model_dir, "ddpg_model.zip") # Final model for benchmarking
            model_stage2.save(final_cl_model_path)
            train_env_stage2.close()
            
            total_cl_training_time = (end_time_stage1 - start_time_stage1) + (end_time_stage2 - start_time_stage2)
            training_times_by_reward[reward_name] = total_cl_training_time
            print(f"-- Finished Curriculum Learning. Model saved in {cl_model_dir} --")

        else: # Existing reward functions
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
            
            if "DDPG" in training_times:
                training_times_by_reward[reward_name] = training_times["DDPG"]
            
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
        
        if "DDPG" in benchmark_results:
            overall_results[reward_name] = benchmark_results["DDPG"]
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
