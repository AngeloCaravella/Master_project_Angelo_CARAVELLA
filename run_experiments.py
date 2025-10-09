
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import gymnasium as gym
from gymnasium.envs.registration import registry
import time
import torch
import traceback
import json
import inspect
import random
import subprocess
from collections import defaultdict
from glob import glob
from typing import List, Dict, Any, Tuple, Callable, Optional

# --- Importazioni dalla libreria custom ev2gym ---
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible, ChargeAsLateAsPossible, RoundRobin
from ev2gym.baselines.pulp_mpc import OnlineMPC_Solver, ApproximateExplicitMPC, ApproximateExplicitMPC_NN, OptimalOfflineSolver
from ev2gym.baselines.cvxpy_mpc_quadratic import OnlineMPC_Solver_Quadratic
from ev2gym.rl_agent.custom_algorithms import CustomDDPG
from ev2gym.utilities.per_buffer import PrioritizedReplayBuffer
from ev2gym.rl_agent import reward as reward_module
from ev2gym.rl_agent.state import V2G_profit_max_loads

# --- Importazioni da librerie di RL ---
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from sb3_contrib import TQC, TRPO, ARS
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from matplotlib.patches import Patch
from stable_baselines3.common.vec_env import DummyVecEnv

# =====================================================================================
# --- CLASSI WRAPPER E AMBIENTI (INVARIATE) ---
# =====================================================================================
class CompatibilityWrapper(gym.Wrapper):
    def __init__(self, env, target_obs_shape, target_action_shape):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=target_obs_shape, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=target_action_shape, dtype=np.float64)

    def _pad_observation(self, obs):
        padded_obs = np.zeros(self.observation_space.shape, dtype=np.float64)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_observation(obs), info

    def step(self, action):
        action_size_needed = self.env.action_space.shape[0]
        sliced_action = action[:action_size_needed]
        obs, reward, terminated, truncated, info = self.env.step(sliced_action)
        return self._pad_observation(obs), reward, terminated, truncated, info

class MultiScenarioEnv(gym.Env):
    def __init__(self, config_files, reward_function, state_function):
        super(MultiScenarioEnv, self).__init__()
        self.config_files = config_files
        self.reward_function = reward_function
        self.state_function = state_function
        self.current_env = None
        max_obs_shape, max_action_shape = 0, 0
        for config in self.config_files:
            temp_env = EV2Gym(config_file=config, reward_function=reward_function, state_function=state_function)
            max_obs_shape = max(max_obs_shape, temp_env.observation_space.shape[0])
            max_action_shape = max(max_action_shape, temp_env.action_space.shape[0])
            temp_env.close()
        self.max_obs_shape = (max_obs_shape,)
        self.max_action_shape = (max_action_shape,)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.max_action_shape, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.max_obs_shape, dtype=np.float64)

    def _pad_observation(self, obs):
        padded_obs = np.zeros(self.max_obs_shape, dtype=np.float64)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def reset(self, *, seed=None, options=None):
        if self.current_env: self.current_env.close()
        selected_config = random.choice(self.config_files)
        self.current_env = EV2Gym(config_file=selected_config, generate_rnd_game=True, reward_function=self.reward_function, state_function=self.state_function)
        obs, info = self.current_env.reset(seed=seed, options=options)
        return self._pad_observation(obs), info

    def step(self, action):
        if self.current_env is None: raise RuntimeError("reset() must be called before step().")
        action_size_needed = self.current_env.action_space.shape[0]
        sliced_action = action[:action_size_needed]
        obs, reward, terminated, truncated, info = self.current_env.step(sliced_action)
        return self._pad_observation(obs), reward, terminated, truncated, info

    def close(self):
        if self.current_env: self.current_env.close()

class CurriculumEnv(gym.Env):
    def __init__(self, config_files, reward_function, state_function, steps_per_level):
        super(CurriculumEnv, self).__init__()
        self.config_files = config_files
        self.reward_function = reward_function
        self.state_function = state_function
        self.steps_per_level = steps_per_level
        self.current_level = 0
        self.steps_at_current_level = 0
        self.current_env = None
        
        max_obs_shape, max_action_shape = 0, 0
        for config in self.config_files:
            temp_env = EV2Gym(config_file=config, reward_function=reward_function, state_function=state_function)
            max_obs_shape = max(max_obs_shape, temp_env.observation_space.shape[0])
            max_action_shape = max(max_action_shape, temp_env.action_space.shape[0])
            temp_env.close()
        self.max_obs_shape = (max_obs_shape,)
        self.max_action_shape = (max_action_shape,)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.max_action_shape, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.max_obs_shape, dtype=np.float64)

    def _pad_observation(self, obs):
        padded_obs = np.zeros(self.max_obs_shape, dtype=np.float64)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def _create_env_for_level(self, level):
        if self.current_env:
            self.current_env.close()
        config_file = self.config_files[level]
        self.current_env = EV2Gym(config_file=config_file, generate_rnd_game=True, reward_function=self.reward_function, state_function=self.state_function)
        return self.current_env

    def reset(self, *, seed=None, options=None):
        if self.current_env is None:
            self._create_env_for_level(self.current_level)
        
        obs, info = self.current_env.reset(seed=seed, options=options)
        return self._pad_observation(obs), info

    def step(self, action):
        if self.current_env is None:
            raise RuntimeError("reset() must be called before step().")
        
        action_size_needed = self.current_env.action_space.shape[0]
        sliced_action = action[:action_size_needed]
        obs, reward, terminated, truncated, info = self.current_env.step(sliced_action)

        self.steps_at_current_level += 1
        if self.steps_at_current_level >= self.steps_per_level:
            if self.current_level < len(self.config_files) - 1:
                self.current_level += 1
                print(f"\n--- Curriculum Learning: Passaggio al livello {self.current_level + 1}/{len(self.config_files)} ---")
                print(f"--- Caricamento scenario: {os.path.basename(self.config_files[self.current_level])} ---")
                self._create_env_for_level(self.current_level)
                self.steps_at_current_level = 0
            else:
                print("\n--- Curriculum Learning: Completato l'ultimo livello del curriculum ---")
                terminated = True

        return self._pad_observation(obs), reward, terminated, truncated, info

    def close(self):
        if self.current_env:
            self.current_env.close()

class ShuffledMultiScenarioEnv(gym.Env):
    def __init__(self, config_files, reward_function, state_function):
        super(ShuffledMultiScenarioEnv, self).__init__()
        self.config_files = config_files
        self.reward_function = reward_function
        self.state_function = state_function
        self.current_env = None
        
        max_obs_shape, max_action_shape = 0, 0
        for config in self.config_files:
            temp_env = EV2Gym(config_file=config, reward_function=reward_function, state_function=state_function)
            max_obs_shape = max(max_obs_shape, temp_env.observation_space.shape[0])
            max_action_shape = max(max_action_shape, temp_env.action_space.shape[0])
            temp_env.close()
        self.max_obs_shape = (max_obs_shape,)
        self.max_action_shape = (max_action_shape,)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.max_action_shape, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.max_obs_shape, dtype=np.float64)

        self.shuffled_scenarios = self.config_files.copy()
        random.shuffle(self.shuffled_scenarios)
        print(f"\n--- Avvio Epoca 1 con {len(self.shuffled_scenarios)} scenari in ordine casuale. ---\n")

    def _pad_observation(self, obs):
        padded_obs = np.zeros(self.max_obs_shape, dtype=np.float64)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def reset(self, *, seed=None, options=None):
        if not self.shuffled_scenarios:
            print("\n--- Epoca completata. Rimescolo gli scenari per la nuova epoca. ---\n")
            self.shuffled_scenarios = self.config_files.copy()
            random.shuffle(self.shuffled_scenarios)
        
        selected_config = self.shuffled_scenarios.pop(0)

        if self.current_env: self.current_env.close()
        self.current_env = EV2Gym(config_file=selected_config, generate_rnd_game=True, reward_function=self.reward_function, state_function=self.state_function)
        obs, info = self.current_env.reset(seed=seed, options=options)
        return self._pad_observation(obs), info

    def step(self, action):
        if self.current_env is None: raise RuntimeError("reset() must be called before step().")
        action_size_needed = self.current_env.action_space.shape[0]
        sliced_action = action[:action_size_needed]
        obs, reward, terminated, truncated, info = self.current_env.step(sliced_action)
        return self._pad_observation(obs), reward, terminated, truncated, info

    def close(self):
        if self.current_env: self.current_env.close()

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, check_freq: int = 1000, verbose: int = 1):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps, self.check_freq = total_timesteps, check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            progress = self.num_timesteps / self.total_timesteps
            elapsed_time = time.time() - self.start_time
            eta = elapsed_time * (1 - progress) / progress if progress > 0 else 0
            print(f"Timesteps: {self.num_timesteps}/{self.total_timesteps} ({progress:.2%}) | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}", flush=True)
        return True

class TrainingPlotCallback(BaseCallback):
    def __init__(self, model_name: str, save_plot: bool = True, verbose: int = 0):
        super(TrainingPlotCallback, self).__init__(verbose)
        self.model_name = model_name
        self.save_plot = save_plot
        self.rewards = []
        self.steps = []

    def _on_step(self) -> bool:
        if 'episode' in self.locals['infos'][0]:
            ep_reward = self.locals['infos'][0]['episode']['r']
            self.rewards.append(ep_reward)
            self.steps.append(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.save_plot:
            print(f"Plotting training results for {self.model_name}...")
            
            if not self.steps:
                print("No episode rewards were logged. Cannot create a plot.")
                return

            rewards_df = pd.DataFrame({'steps': self.steps, 'rewards': self.rewards})
            rolling_avg = rewards_df.rewards.rolling(window=max(1, len(self.rewards) // 10)).mean()

            plt.figure(figsize=(10, 5))
            plt.plot(rewards_df.steps, rewards_df.rewards, 'b-', linewidth=0.5, alpha=0.4, label='Episodic Return')
            plt.plot(rewards_df.steps, rolling_avg, 'r-', linewidth=2, label='Rolling Average')
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title(f"Training Progress for {self.model_name}")
            plt.grid(True)
            plt.legend()
            
            plot_dir = "training_plots"
            os.makedirs(plot_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{timestamp}.png"
            plt.savefig(os.path.join(plot_dir, filename))
            plt.close()
            print(f"Training plot saved to {os.path.join(plot_dir, filename)}")

# =====================================================================================
# --- FUNZIONI DI PLOTTING (INVARIATE) ---
# =====================================================================================
def get_color_map_and_legend(algorithms_to_plot):
    """Generates a unique color map and legend for each algorithm."""
    # Define a list of visually distinct colors for the plots.
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', 
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94' 
    ]
    
    # Create a mapping from each algorithm to its own name (which acts as its category).
    algo_categories = {algo: algo for algo in algorithms_to_plot}
    
    # Create a mapping from each algorithm name to a unique color.
    color_map = {algo: distinct_colors[i % len(distinct_colors)] for i, algo in enumerate(algorithms_to_plot)}
    
    # Add a default color for categories not explicitly defined
    color_map["default"] = "#cccccc" # Light grey for default

    # Create legend elements for each individual algorithm.
    legend_elements = [Patch(facecolor=color_map[algo], edgecolor='black', label=algo) for algo in algorithms_to_plot]
    
    return algo_categories, color_map, legend_elements

def plot_performance_metrics(stats_collection, save_path, scenario_name, algorithms_to_plot):
    if not stats_collection: return
    # Removed 'battery_degradation' from the metrics map
    metrics_map = {
        'total_profits': 'Total Profit (€)', 'average_user_satisfaction': 'Average User Satisfaction (%)',
        'peak_transformer_loading_pct': 'Peak Transformer Loading (%)' 
    }
    model_names = [name for name in algorithms_to_plot if name in stats_collection]
    if not model_names: return
    algo_categories, category_colors, legend_elements = get_color_map_and_legend(model_names)
    # Changed subplot layout to 1x3
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)); axes = axes.flatten()
    fig.suptitle(f'Aggregated Performance Metrics - Scenario: {scenario_name}', fontsize=22)
    for i, (metric, title) in enumerate(metrics_map.items()):
        ax = axes[i]
        means = [stats_collection[name]['mean'].get(metric, 0) for name in model_names]
        stds = [stats_collection[name]['std'].get(metric, 0) for name in model_names]
        if ('satisfaction' in metric or 'degradation' in metric) and 'loading' not in metric:
            means = [v * 100 for v in means]
            stds = [v * 100 for v in stds]
        colors = [category_colors.get(algo_categories.get(name, "default"), category_colors["default"]) for name in model_names]
        ax.bar(model_names, means, yerr=stds, color=colors, capsize=5)
        ax.set_title(title, fontsize=14); ax.tick_params(axis='x', rotation=45); ax.grid(True, linestyle='--', alpha=0.6)
        if '(%)' in title: ax.set_ylim(0, max(105, (max(means) if means else 0) * 1.1))
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, f"performance_summary_{scenario_name}.png")); plt.close(fig)

def plot_ev_presence(save_path, scenario_name, ev_counts, timescale):
    """Plots the number of active EVs over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    time_hours = np.arange(len(ev_counts)) * timescale / 60
    ax.plot(time_hours, ev_counts, label='Number of EVs')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Number of Active EVs")
    ax.set_title(f"Number of Active EVs over Time - Scenario: {scenario_name}")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"ev_presence_{scenario_name}.png"))
    plt.close(fig)

def plot_average_soc_over_time(save_path, scenario_name, soc_data, timescale):
    """Plots the average SoC over time for each algorithm."""
    if not soc_data:
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get the color mapping
    algorithms_to_plot = list(soc_data.keys())
    algo_categories, category_colors, legend_elements = get_color_map_and_legend(algorithms_to_plot)

    for name, runs in soc_data.items():
        if not runs:
            continue
        
        # Find the length of the shortest run to align arrays
        min_len = min(len(run) for run in runs)
        aligned_runs = [run[:min_len] for run in runs]
        
        mean_soc = np.mean(aligned_runs, axis=0)
        std_soc = np.std(aligned_runs, axis=0)
        
        time_hours = np.arange(min_len) * timescale / 60
        color = category_colors.get(algo_categories.get(name, "default"), category_colors["default"])
        
        ax.plot(time_hours, mean_soc, label=name, color=color, linewidth=2)
        ax.fill_between(time_hours, mean_soc - std_soc, mean_soc + std_soc, color=color, alpha=0.2)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Average State of Charge (SoC)")
    ax.set_title(f"Average SoC Over Time - Scenario: {scenario_name}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Algorithms")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"average_soc_over_time_{scenario_name}.png"))
    plt.close(fig)

def plot_tradeoff_scatter(stats_collection, save_path, scenario_name, algorithms_to_plot):
    """Plots a scatter matrix to show trade-offs between key metrics."""
    if not stats_collection:
        return

    metrics_to_compare = {
        'total_profits': 'Total Profit (€)',
        'peak_transformer_loading_pct': 'Peak Load (%)',
        'average_user_satisfaction': 'User Satisfaction (%)',
        'battery_degradation': 'Degradation (%)'
    }

    model_names = [name for name in algorithms_to_plot if name in stats_collection]
    if not model_names:
        return

    df_data = []
    for name in model_names:
        row = {'Algorithm': name}
        for key, label in metrics_to_compare.items():
            value = stats_collection[name]['mean'].get(key, 0)
            # Scale satisfaction and degradation to %
            if 'satisfaction' in key or 'degradation' in key:
                value *= 100
            row[label] = value
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    if df.empty:
        return

    num_metrics = len(metrics_to_compare)
    fig, axes = plt.subplots(num_metrics, num_metrics, figsize=(18, 18))
    fig.suptitle(f'Performance Trade-offs - Scenario: {scenario_name}', fontsize=22, y=0.95)

    metric_labels = list(metrics_to_compare.values())
    algo_categories, category_colors, _ = get_color_map_and_legend(model_names)
    colors = [category_colors.get(algo_categories.get(name, "default"), category_colors["default"]) for name in df['Algorithm']]

    for i in range(num_metrics):
        for j in range(num_metrics):
            ax = axes[i, j]
            x_metric = metric_labels[j]
            y_metric = metric_labels[i]

            if i == j:
                # On the diagonal, plot a histogram of the metric
                ax.hist(df[x_metric], color='#bdc3c7')
            else:
                ax.scatter(df[x_metric], df[y_metric], c=colors, s=100, alpha=0.8)
                # Annotate points
                for k, txt in enumerate(df['Algorithm']):
                    ax.annotate(txt, (df[x_metric][k], df[y_metric][k]), xytext=(5,5), textcoords='offset points', fontsize=8, alpha=0.7)

            # Set labels
            if i == num_metrics - 1:
                ax.set_xlabel(x_metric, fontsize=12)
            if j == 0:
                ax.set_ylabel(y_metric, fontsize=12)
            
            ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(save_path, f"tradeoff_scatter_matrix_{scenario_name}.png"))
    plt.close(fig)

def plot_overload_composition(save_path, scenario_name, algorithm_name, power_data, timescale):
    """Plots a stacked area chart of power composition causing transformer load."""
    
    ev_load = power_data.get('ev_load', [])
    inflexible_load = power_data.get('inflexible_load', [])
    solar_power = power_data.get('solar_power', [])
    limit = power_data.get('transformer_limit')

    if len(ev_load) == 0 or len(inflexible_load) == 0 or len(solar_power) == 0 or limit is None:
        print(f"Skipping overload plot for {algorithm_name} due to missing data.")
        return

    # Ensure all arrays have the same length
    min_len = min(len(ev_load), len(inflexible_load), len(solar_power))
    time_hours = np.arange(min_len) * timescale / 60
    
    ev_load = ev_load[:min_len]
    inflexible_load = inflexible_load[:min_len]
    solar_power = solar_power[:min_len]

    gross_load = ev_load + inflexible_load
    net_load = gross_load - solar_power

    fig, ax = plt.subplots(figsize=(18, 9))
    
    # Stacked Area for loads
    ax.stackplot(time_hours, inflexible_load, ev_load, 
                 labels=['Inflexible Loads', 'EV Charging'], 
                 colors=['#3498db', '#9b59b6'], alpha=0.7)

    # Line for solar power (negative load)
    ax.fill_between(time_hours, 0, -solar_power, color='#2ecc71', alpha=0.6, label='Solar Generation')

    # Line for Net Load
    ax.plot(time_hours, net_load, label='Net Load', color='#e74c3c', linewidth=2.5)

    # Line for Transformer Limit
    limit_series = np.full(min_len, limit) if np.isscalar(limit) else limit[:min_len]
    ax.plot(time_hours, limit_series, color='black', linestyle='--', linewidth=2, label=f'Transformer Limit')

    # Fill area for overload
    ax.fill_between(time_hours, net_load, limit_series, where=net_load > limit_series, 
                    facecolor='red', alpha=0.5, interpolate=True, label='Overload')

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f'Transformer Load Composition - {scenario_name} - Algorithm: {algorithm_name}', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=min(0, np.min(-solar_power)) * 1.1)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    # Sanitize algorithm name for filename
    safe_algo_name = "".join(c for c in algorithm_name if c.isalnum() or c in ('_', '-')).rstrip()
    plt.savefig(os.path.join(save_path, f"overload_composition_{scenario_name}_{safe_algo_name}.png"))
    plt.close(fig)

def plot_electricity_prices(save_path, scenario_name, charge_prices, discharge_prices, timescale):
    """Plots the electricity charge and discharge prices over time."""
    if charge_prices is None or discharge_prices is None:
        return

    # Assuming prices are the same for all stations, take the first one.
    charge_price_series = charge_prices[0]
    discharge_price_series = discharge_prices[0]
    
    min_len = min(len(charge_price_series), len(discharge_price_series))
    time_hours = np.arange(min_len) * timescale / 60

    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(time_hours, charge_price_series[:min_len], label='Charge Price', color='#e74c3c', linewidth=2)
    ax.plot(time_hours, discharge_price_series[:min_len], label='Discharge Price', color='#2ecc71', linewidth=2, linestyle='--')

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Price (€/kWh)") # Assuming the unit is €/kWh
    ax.set_title(f'Electricity Prices - Scenario: {scenario_name}', fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"electricity_prices_{scenario_name}.png"))
    plt.close(fig)

def plot_degradation_components(save_path, scenario_name, degradation_data, algorithms_to_plot):
    """Plots the components of battery degradation against average SoC."""
    if not degradation_data:
        return

    df = pd.DataFrame(degradation_data)
    if df.empty:
        return

    # Get the unique color map for algorithms
    _, color_map, _ = get_color_map_and_legend(algorithms_to_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
    fig.suptitle(f'Degradation Analysis - Scenario: {scenario_name}', fontsize=20)

    # Plot 1: Calendar Degradation
    ax1.set_title('Calendar Degradation vs. Average SoC', fontsize=16)
    for name, group in df.groupby('algorithm'):
        ax1.scatter(group['avg_soc'], group['d_cal'], 
                    label=name, color=color_map.get(name, '#000000'), 
                    alpha=0.6, s=50)
    ax1.set_xlabel('Average SoC during parking session')
    ax1.set_ylabel('Capacity Loss Fraction (d_cal)')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Cyclic Degradation
    ax2.set_title('Cyclic Degradation vs. Average SoC', fontsize=16)
    for name, group in df.groupby('algorithm'):
        ax2.scatter(group['avg_soc'], group['d_cyc'], 
                    label=name, color=color_map.get(name, '#000000'), 
                    alpha=0.6, s=50)
    ax2.set_xlabel('Average SoC during parking session')
    ax2.set_ylabel('Capacity Loss Fraction (d_cyc)')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Create a single legend for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(algorithms_to_plot), 6), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, f"degradation_components_{scenario_name}.png"))
    plt.close(fig)






# =====================================================================================
# --- FUNZIONE PER GENERARE OUTPUT RIASSUNTIVI (MODIFICATA) ---
# =====================================================================================
def generate_summary_outputs(stats_collection, save_path, scenario_name, num_simulations):
    """Generates and saves an image and a CSV with the aggregated results."""
    
    column_mapping = {
        'total_profits': ('Profits/Costs (€)', 1),
        'average_user_satisfaction': ('User Sat. (%)', 100),
        'total_energy_charged': ('Energy Ch. (kWh)', 1),
        'total_energy_discharged': ('Energy Disch. (kWh)', 1),
        'transformer_overload_kwh': ('Tr. Ov. (kWh)', 1),
        'battery_degradation': ('Total Qˡᵒˢᵗ (x10⁻³)', 1000),
        'battery_degradation_calendar': ('Σ dᶜᵃˡ (x10⁻³)', 1000),
        'battery_degradation_cyclic': ('Σ dᶜʸᶜ (x10⁻³)', 1000),
        'execution_time': ('Exec. Time (s)', 1),
        'total_reward': ('Reward (x10³)', 0.001)
    }
    
    summary_data = []
    for name, data in stats_collection.items():
        row = {'Algorithm': name}
        for key, (col_name, scale) in column_mapping.items():
            mean = data['mean'].get(key, 0)
            std = data['std'].get(key, 0)
            
            mean_scaled = mean * scale
            std_scaled = std * scale
            
            if num_simulations > 1:
                if 'Time' in col_name:
                    row[col_name] = f"{mean_scaled:.2f} ± {std_scaled:.2f}"
                else:
                    row[col_name] = f"{mean_scaled:.1f} ± {std_scaled:.1f}"
            else:
                if 'Time' in col_name:
                    row[col_name] = f"{mean_scaled:.2f}"
                else:
                    row[col_name] = f"{mean_scaled:.1f}"
        summary_data.append(row)
        
    if not summary_data:
        print("No aggregated data to save.")
        return

    df_summary = pd.DataFrame(summary_data).set_index('Algorithm')
    
    csv_path = os.path.join(save_path, f"summary_results_{scenario_name}.csv")
    df_summary.to_csv(csv_path)
    print(f"Aggregated results table saved to: {csv_path}")

    fig, ax = plt.subplots(figsize=(22, 1 + len(df_summary) * 0.5))
    ax.axis('tight'); ax.axis('off')
    
    table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns, rowLabels=df_summary.index,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 1.3)

    for (row, col), cell in table.get_celld().items():
        if (row == 0 or col == -1):
            cell.set_text_props(weight='bold')
        if row % 2 == 1 and row > 0:
            cell.set_facecolor('#f2f2f2')

    fig.suptitle(f'Aggregated Results - Scenario: {scenario_name} (N={num_simulations} simulations)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    img_path = os.path.join(save_path, f"summary_table_{scenario_name}.png")
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Aggregated results table image saved to: {img_path}")

# =====================================================================================
# --- FUNZIONE DI BENCHMARK CON SANITY CHECK ROBUSTO ---
# =====================================================================================
def run_benchmark(config_files, reward_func, algorithms_to_run, num_simulations, model_dir, is_multi_scenario, price_data_file=None):
    all_scenario_stats = {}
    overall_save_path = f'./results/benchmark_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(overall_save_path, exist_ok=True)

    max_obs_shape, max_action_shape = (0,), (0,)
    if is_multi_scenario:
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            max_obs_shape = tuple(metadata["observation_space_shape"])
            max_action_shape = tuple(metadata["action_space_shape"])
            print(f"Wrapper shapes loaded from metadata: OBS={max_obs_shape}, ACT={max_action_shape}")
        else:
            print("WARNING: 'model_metadata.json' file not found.")
            temp_env = MultiScenarioEnv(config_files, reward_func, V2G_profit_max_loads)
            max_obs_shape, max_action_shape = temp_env.observation_space.shape, temp_env.action_space.shape
            temp_env.close()

    for config_file in config_files:
        scenario_name = os.path.basename(config_file).replace(".yaml", "")
        print(f"\n\n{'='*80}\nSTARTING BENCHMARK FOR SCENARIO: {scenario_name}\n{'='*80}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        tr_capacity = config.get('transformer_capacity_kva', 0) or config.get('transformer', {}).get('max_power', 0)
        num_cs = config.get('number_of_charging_stations', 0)
        max_power_cs = 0
        if 'charging_stations' in config and config['charging_stations']:
             max_power_cs = config['charging_stations'][0].get('max_power', 0)
        elif 'charging_station' in config:
            cs_config = config['charging_station']
            voltage = cs_config.get('voltage', 0); current = cs_config.get('max_charge_current', 0); phases = cs_config.get('phases', 1)
            if voltage > 0 and current > 0:
                max_power_cs = (voltage * current * (np.sqrt(3) if phases == 3 else 1)) / 1000.0

        if (max_theoretical_load := num_cs * max_power_cs) <= tr_capacity and tr_capacity > 0:
            print("*"*80)
            print(f"WARNING: Transformer overload is IMPOSSIBLE in this scenario.")
            print(f"  - Transformer Capacity: {tr_capacity:.2f} kVA")
            print(f"  - Max Theoretical Load (N_stations * P_max): {max_theoretical_load:.2f} kW")
            print("  - To test overload management, increase the number of stations or decrease transformer capacity.")
            print("*"*80)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        scenario_save_path = os.path.join(overall_save_path, scenario_name); os.makedirs(scenario_save_path, exist_ok=True)

        all_sim_stats = defaultdict(lambda: defaultdict(list))
        soc_over_time_by_algo = defaultdict(list)
        degradation_data_points = [] # To store data for the new degradation plot

        # --- Data collection for EV presence plot ---
        print("\n--- Collecting EV presence data for the scenario ---")
        presence_env = EV2Gym(config_file=config_file, generate_rnd_game=True)
        ev_counts_over_time = []
        done = False
        while not done:
            num_evs = sum(1 for cs in presence_env.charging_stations for ev in cs.evs_connected if ev is not None)
            ev_counts_over_time.append(num_evs)
            _, _, done, _, _ = presence_env.step(np.zeros(presence_env.action_space.shape[0]))
        timescale = presence_env.timescale
        charge_prices = presence_env.charge_prices
        discharge_prices = presence_env.discharge_prices
        presence_env.close()

        for sim_num in range(num_simulations):
            print(f"\n--- Simulation {sim_num + 1}/{num_simulations} ---")
            env_replay = EV2Gym(config_file=config_file, generate_rnd_game=True, save_replay=True, price_data_file=price_data_file)
            replay_path = f"replay/replay_{env_replay.sim_name}.pkl"
            while not env_replay.step(np.zeros(env_replay.action_space.shape[0]))[2]: pass
            env_replay.close()

            eval_env_id = f'eval-env-{scenario_name}-{sim_num}'
            if eval_env_id in registry: del registry[eval_env_id]
            gym.register(id=eval_env_id, entry_point='ev2gym.models.ev2gym_env:EV2Gym', kwargs={'config_file': config_file, 'generate_rnd_game': True, 'load_from_replay_path': replay_path, 'reward_function': reward_func, 'state_function': V2G_profit_max_loads, 'price_data_file': price_data_file})

            for name, (algorithm_class, rl_class, kwargs) in algorithms_to_run.items():
                print(f"+ Running: {name}")
                try:
                    env_instance = gym.make(eval_env_id)
                    is_rl_model = rl_class is not None
                    if is_rl_model:
                        if is_multi_scenario: env_instance = CompatibilityWrapper(env_instance, max_obs_shape, max_action_shape)
                        model_path = os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip')
                        if not os.path.exists(model_path): print(f"!!! Model {name} not found. Skipping."); continue
                        model = rl_class.load(model_path, env=env_instance, device=device)
                    else:
                        model = algorithm_class(env=env_instance.unwrapped, **kwargs)

                    obs, _ = env_instance.reset()
                    done = False
                    start_time = time.time()
                    
                    current_run_soc_over_time = []
                    while not done:
                        action = model.predict(obs, deterministic=True)[0] if is_rl_model else model.get_action(env_instance.unwrapped)
                        obs, _, done, _, _ = env_instance.step(action)
                        
                        # Calculate and record average SoC for the current step
                        connected_evs = [ev for cs in env_instance.unwrapped.charging_stations for ev in cs.evs_connected if ev is not None]
                        if connected_evs:
                            avg_soc = np.mean([ev.current_capacity / ev.battery_capacity for ev in connected_evs])
                            current_run_soc_over_time.append(avg_soc)
                        else:
                            current_run_soc_over_time.append(0) # Append 0 if no EVs are connected

                    soc_over_time_by_algo[name].append(current_run_soc_over_time)

                    # Collect data for overload composition plot on the first run
                    if sim_num == 0:
                        power_data = {
                            'ev_load': np.sum(env_instance.unwrapped.cs_power, axis=0),
                            'inflexible_load': np.sum(env_instance.unwrapped.tr_inflexible_loads, axis=0),
                            'solar_power': np.sum(env_instance.unwrapped.tr_solar_power, axis=0),
                            'transformer_limit': env_instance.unwrapped.transformers[0].max_power
                        }
                        plot_overload_composition(scenario_save_path, scenario_name, name, power_data, env_instance.unwrapped.timescale)

                    execution_time = time.time() - start_time
                    
                    stats = env_instance.unwrapped.stats
                    stats['execution_time'] = execution_time
                    departed_evs = env_instance.unwrapped.departed_evs
                    if departed_evs:
                        for ev in departed_evs:
                            ev.get_battery_degradation()
                        stats['battery_degradation_calendar'] = np.mean([ev.calendar_loss for ev in departed_evs])
                        stats['battery_degradation_cyclic'] = np.mean([ev.cyclic_loss for ev in departed_evs])
                        stats['battery_degradation'] = stats['battery_degradation_calendar'] + stats['battery_degradation_cyclic']

                        # Collect data for the new degradation component plot
                        for ev in departed_evs:
                            if ev.historic_soc:
                                avg_soc = np.mean(ev.historic_soc)
                                degradation_data_points.append({
                                    'algorithm': name,
                                    'avg_soc': avg_soc,
                                    'd_cal': ev.calendar_loss,
                                    'd_cyc': ev.cyclic_loss
                                })


                    for metric, value in stats.items():
                        all_sim_stats[name][metric].append(value)
                    
                    env_instance.close()
                except Exception as e:
                    print(f"!!! ERROR with '{name}': {e}. Skipping."); traceback.print_exc()
            
            if os.path.exists(replay_path): os.remove(replay_path)

        if all_sim_stats:
            aggregated_stats = defaultdict(dict)
            for name, metrics in all_sim_stats.items():
                aggregated_stats[name]['mean'] = {metric: np.mean(values) for metric, values in metrics.items()}
                aggregated_stats[name]['std'] = {metric: np.std(values) for metric, values in metrics.items()}
            
            all_scenario_stats[scenario_name] = aggregated_stats
            
            # --- Generate all plots and summaries ---
            print("\n--- Generating output plots and summaries ---")
            plot_performance_metrics(aggregated_stats, scenario_save_path, scenario_name, list(algorithms_to_run.keys()))
            plot_ev_presence(scenario_save_path, scenario_name, ev_counts_over_time, timescale)
            plot_average_soc_over_time(scenario_save_path, scenario_name, soc_over_time_by_algo, timescale)
            plot_tradeoff_scatter(aggregated_stats, scenario_save_path, scenario_name, list(algorithms_to_run.keys()))
            plot_electricity_prices(scenario_save_path, scenario_name, charge_prices, discharge_prices, timescale)
            plot_degradation_components(scenario_save_path, scenario_name, degradation_data_points, list(algorithms_to_run.keys()))
            generate_summary_outputs(aggregated_stats, scenario_save_path, scenario_name, num_simulations)

    print(f"\n--- Benchmark finished. Results saved in: {overall_save_path} ---")

# =====================================================================================
# --- BLOCCO PRINCIPALE (invariato) ---
# =====================================================================================
def calculate_max_cs(config_path: str) -> int:
    """Calcola il numero massimo di stazioni di ricarica tra tutti gli scenari."""
    all_scenarios_for_cs = glob(os.path.join(config_path, "*.yaml"))
    max_cs = 0
    if not all_scenarios_for_cs:
        print(f"ATTENZIONE: Nessun file di scenario trovato in {config_path}, MAX_CS impostato a 10 di default.")
        return 10
    else:
        for scenario_file in all_scenarios_for_cs:
            with open(scenario_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if 'number_of_charging_stations' in config:
                    max_cs = max(max_cs, config['number_of_charging_stations'])
    if max_cs == 0:
        raise ValueError("Impossibile determinare il numero massimo di stazioni di ricarica.")
    return max_cs


def get_algorithms(max_cs: int, is_thesis_mode: bool, mpc_type: str = 'linear') -> Dict[str, Tuple[Any, Any, Dict]]:
    """
    Definisce e restituisce un dizionario di algoritmi disponibili per l'esecuzione.

    Args:
        max_cs (int): Il numero massimo di stazioni di ricarica, necessario per alcuni modelli.
        is_thesis_mode (bool): Se True, restituisce solo il sottoinsieme di algoritmi usati per la tesi.
        mpc_type (str): Tipo di MPC da includere ('linear' o 'quadratic').

    Returns:
        Dict[str, Tuple[Any, Any, Dict]]: Un dizionario dove la chiave è il nome dell'algoritmo
        e il valore è una tupla contenente:
        - La classe dell'algoritmo (se non è un modello RL).
        - La classe del modello RL di Stable Baselines (se applicabile).
        - Un dizionario di argomenti (kwargs) per l'inizializzazione.
    """
    # Algoritmi di base: euristiche, modelli RL e il solver ottimale offline
    base_algorithms = {
        "AFAP": (ChargeAsFastAsPossible, None, {}), 
        "ALAP": (ChargeAsLateAsPossible, None, {}), 
        "RR": (RoundRobin, None, {}),
        "SAC": (None, SAC, {}), 
        "PPO": (None, PPO, {}), 
        "A2C": (None, A2C, {}), 
        "TD3": (None, TD3, {}), 
        "DDPG": (None, DDPG, {}),
        "DDPG+PER": (None, CustomDDPG, {'replay_buffer_class': PrioritizedReplayBuffer}),
        "TQC": (None, TQC, {}), 
        "TRPO": (None, TRPO, {}), 
        "ARS": (None, ARS, {}),
        "Optimal_Offline": (OptimalOfflineSolver, None, {}),  # <-- AGGIUNTO: Il solver ottimale
    }
    
    # Algoritmi MPC Online, selezionabili tramite il parametro mpc_type
    mpc_algorithms = {
        'linear': {
            "Online_MPC_Profit_Max": (OnlineMPC_Solver, None, {
                'prediction_horizon': 25, 
                'control_horizon': 'half'
            }),
        },
        'quadratic': {
            "Online_MPC_Quadratic": (OnlineMPC_Solver_Quadratic, None, {'control_horizon': 5}),
        }
    }

    # Algoritmi MPC Espliciti Approssimati
    approx_mpc = {
        "Approx_Explicit_MPC": (ApproximateExplicitMPC, None, {'control_horizon': 5, 'max_cs': max_cs}),
        "Approx_Explicit_MPC_NN": (ApproximateExplicitMPC_NN, None, {'control_horizon': 5, 'max_cs': max_cs}),
    }
    
    # Unisce tutti i dizionari per creare l'elenco completo degli algoritmi
    ALL_ALGORITHMS = {**base_algorithms, **mpc_algorithms.get(mpc_type, {}), **approx_mpc}
    
    # Definisce il sottoinsieme di algoritmi da usare in "modalità tesi"
    THESIS_ALGORITHMS_BASE = [
        "AFAP", 
        "ALAP", 
        "RR", 
        "SAC", 
        "DDPG+PER", 
        "TQC", 
        "Online_MPC_Profit_Max", 
        "Approx_Explicit_MPC", 
        "Approx_Explicit_MPC_NN",
        "Optimal_Offline"  # <-- AGGIUNTO: Il solver ottimale anche qui per il confronto
    ]
                              
    # Filtra l'elenco completo per ottenere solo gli algoritmi della modalità tesi
    THESIS_ALGORITHMS = {k: v for k, v in ALL_ALGORITHMS.items() if k in THESIS_ALGORITHMS_BASE}
    
    # Restituisce l'elenco appropriato in base alla modalità scelta
    return THESIS_ALGORITHMS if is_thesis_mode else ALL_ALGORITHMS


def get_scenarios_to_test(config_path: str) -> List[str]:
    """Ottiene la lista degli scenari disponibili e chiede all'utente quali testare."""
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    print("\nScenari disponibili:")
    for i, s in enumerate(available_scenarios): print(f"{i+1}. {os.path.basename(s)}")
    choices = input(f"Seleziona scenari (es. '1 3', 'tutti') (default: tutti): ").lower() or 'tutti'
    scenarios_to_test = available_scenarios if 'tutti' in choices else [available_scenarios[int(i)-1] for i in choices.split()]
    print(f"Scenari selezionati: {[os.path.basename(s) for s in scenarios_to_test]}")
    return scenarios_to_test


def get_selected_reward_function() -> Callable:
    """Ottiene la lista delle funzioni di reward disponibili e chiede all'utente quale selezionare."""
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    print("\nScegli la funzione di reward:")
    for i, (name, func) in enumerate(available_rewards):
        doc = inspect.getdoc(func); short_doc = (doc.strip().split('\n')[0] if doc else "Nessuna descrizione.")
        print(f"{i + 1}. {name} - {short_doc}")
    reward_choice = int(input(f"Scelta (default 3): ") or 3)
    selected_reward_func = available_rewards[reward_choice - 1][1]
    return selected_reward_func


def get_selected_price_file() -> Optional[str]:
    """Ottiene la lista dei file CSV dei prezzi disponibili e chiede all'utente quale selezionare."""
    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    print("\nSeleziona il file CSV per i prezzi dell'energia:")
    for i, f in enumerate(available_price_files): print(f"{i+1}. {f}")
    price_choice_input = input(f"Scelta (default: Netherlands_day-ahead-2015-2024.csv): ")
    selected_price_file_abs_path = None
    default_price_file_name = "Netherlands_day-ahead-2015-2024.csv"
    if price_choice_input.isdigit() and 1 <= int(price_choice_input) <= len(available_price_files):
        chosen_file_name = available_price_files[int(price_choice_input) - 1]
        selected_price_file_abs_path = os.path.join(price_data_dir, chosen_file_name)
        print(f"File prezzi selezionato: {chosen_file_name}")
    elif price_choice_input == '' or price_choice_input.lower() == default_price_file_name.lower():
        if default_price_file_name in available_price_files:
            selected_price_file_abs_path = os.path.join(price_data_dir, default_price_file_name)
            print(f"File prezzi di default: {default_price_file_name}")
        else:
            print(f"ATTENZIONE: File di prezzo di default '{default_price_file_name}' non trovato.")
    else:
        print(f"Scelta non valida.")
    return selected_price_file_abs_path


def train_rl_models_if_requested(scenarios_to_test: List[str], selected_reward_func: Callable, algorithms_to_run: Dict, is_multi_scenario: bool, model_dir: str, selected_price_file_abs_path: Optional[str], steps_for_training: int, training_mode: str = 'single', curriculum_steps_per_level: int = 10000) -> None:
    """Addestra i modelli RL se richiesto."""
    rl_models_to_run = {k: v for k, v in algorithms_to_run.items() if v[1] is not None}
    mode_str = "Multi-Scenario" if is_multi_scenario else "Single-Domain"
    print(f"--- Addestramento modelli RL in modalità {mode_str} ---")
    scenario_name_for_path = 'multi_scenario' if is_multi_scenario else os.path.basename(scenarios_to_test[0]).replace(".yaml", "")
    train_env_id = f'ev-train-{scenario_name_for_path}'
    if training_mode == 'single':
        if train_env_id in registry: del registry[train_env_id]
        gym.register(id=train_env_id, entry_point='ev2gym.models.ev2gym_env:EV2Gym', kwargs={'config_file': scenarios_to_test[0], 'generate_rnd_game': True, 'reward_function': selected_reward_func, 'state_function': V2G_profit_max_loads, 'price_data_file': selected_price_file_abs_path})
        train_env = gym.make(train_env_id)
        train_env = Monitor(train_env)
    else:
        if training_mode == 'random':
            env_lambda = lambda: Monitor(MultiScenarioEnv(scenarios_to_test, selected_reward_func, V2G_profit_max_loads))
        elif training_mode == 'curriculum':
            env_lambda = lambda: Monitor(CurriculumEnv(scenarios_to_test, selected_reward_func, V2G_profit_max_loads, curriculum_steps_per_level))
        elif training_mode == 'shuffled':
            env_lambda = lambda: Monitor(ShuffledMultiScenarioEnv(scenarios_to_test, selected_reward_func, V2G_profit_max_loads))
        else:
            print(f"ATTENZIONE: Modalità di training '{training_mode}' non riconosciuta. Verrà usata la modalità 'random' di default.")
            env_lambda = lambda: Monitor(MultiScenarioEnv(scenarios_to_test, selected_reward_func, V2G_profit_max_loads))
        train_env = DummyVecEnv([env_lambda])

    for name, (_, rl_class, kwargs) in rl_models_to_run.items():
        print(f"--- Addestramento {name} in modalità {mode_str} ---")
        model = rl_class("MlpPolicy", train_env, verbose=0, device=("cuda" if torch.cuda.is_available() else "cpu"), **kwargs)
        progress_callback = ProgressCallback(total_timesteps=steps_for_training)
        plot_callback = TrainingPlotCallback(model_name=name)
        model.learn(total_timesteps=steps_for_training, callback=[progress_callback, plot_callback])
        model.save(os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip'))

    if is_multi_scenario:
        obs_shape = train_env.observation_space.shape
        act_shape = train_env.action_space.shape
        metadata = {"observation_space_shape": list(obs_shape), "action_space_shape": list(act_shape)}
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadati del modello salvati in {os.path.join(model_dir, 'model_metadata.json')}")
    train_env.close()


def run_fit_battery_if_requested() -> None:
    """Chiede all'utente se eseguire Fit_battery.py e, in caso affermativo, lo esegue."""
    if input("Vuoi eseguire 'Fit_battery.py' per calibrare il modello di degradazione? (s/n, default n): ").lower() == 's':
        print("--- Esecuzione di Fit_battery.py ---")
        try:
            subprocess.run(["python", "Fit_battery.py"], check=True)
            print("--- Fit_battery.py completato. ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERRORE: {e}. Lo script procederà con i parametri esistenti.")
