# --- START OF FILE run_experiments.py ---

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
from ev2gym.baselines import pulp_mpc # Import the module

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
from desired_soc_summary import print_desired_soc_summary

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
                print(f"\n--- Curriculum Learning: Advancing to level {self.current_level + 1}/{len(self.config_files)} ---")
                print(f"--- Loading scenario: {os.path.basename(self.config_files[self.current_level])} ---")
                self._create_env_for_level(self.current_level)
                self.steps_at_current_level = 0
            else:
                print("\n--- Curriculum Learning: Completed the last level of the curriculum ---")
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
        print(f"\n--- Starting Epoch 1 with {len(self.shuffled_scenarios)} scenarios in random order. ---\n")

    def _pad_observation(self, obs):
        padded_obs = np.zeros(self.max_obs_shape, dtype=np.float64)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def reset(self, *, seed=None, options=None):
        if not self.shuffled_scenarios:
            print("\n--- Epoch completed. Shuffling scenarios for the new epoch. ---\n")
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
# --- FUNZIONI DI PLOTTING ---
# =====================================================================================
def get_color_map_and_legend(algorithms_to_plot):
    """Generates a unique color map and legend for each algorithm."""
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', 
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94' 
    ]
    algo_categories = {algo: algo for algo in algorithms_to_plot}
    color_map = {algo: distinct_colors[i % len(distinct_colors)] for i, algo in enumerate(algorithms_to_plot)}
    color_map["default"] = "#cccccc"
    legend_elements = [Patch(facecolor=color_map[algo], edgecolor='black', label=algo) for algo in algorithms_to_plot]
    return algo_categories, color_map, legend_elements

def plot_performance_metrics(stats_collection, save_path, scenario_name, algorithms_to_plot):
    if not stats_collection: return
    metrics_map = {
        'total_profits': 'Total Profit (€)', 'average_user_satisfaction': 'Average User Satisfaction (%)',
        'peak_transformer_loading_pct': 'Peak Transformer Loading (%)' 
    }
    model_names = [name for name in algorithms_to_plot if name in stats_collection]
    if not model_names: return
    algo_categories, category_colors, legend_elements = get_color_map_and_legend(model_names)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8)); axes = axes.flatten()
    fig.suptitle(f'Aggregated Performance Metrics - Scenario: {scenario_name}', fontsize=22)
    for i, (metric, title) in enumerate(metrics_map.items()):
        ax = axes[i]
        means = [stats_collection[name]['mean'].get(metric, 0) for name in model_names]
        stds = [stats_collection[name]['std'].get(metric, 0) for name in model_names]
        if 'satisfaction' in metric:
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
    if not soc_data: return
    fig, ax = plt.subplots(figsize=(15, 8))
    algorithms_to_plot = list(soc_data.keys())
    _, category_colors, _ = get_color_map_and_legend(algorithms_to_plot)
    for name, runs in soc_data.items():
        if not runs: continue
        min_len = min(len(run) for run in runs)
        aligned_runs = [run[:min_len] for run in runs]
        mean_soc = np.mean(aligned_runs, axis=0)
        std_soc = np.std(aligned_runs, axis=0)
        time_hours = np.arange(min_len) * timescale / 60
        color = category_colors.get(name, "#cccccc")
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

def plot_individual_ev_sessions(save_path, scenario_name, algorithm_name, departed_evs, timescale):
    """
    Generates a detailed graph showing the charge profile of each individual EV.
    """
    if not departed_evs:
        print(f"No EVs departed for algorithm {algorithm_name}, cannot generate individual plot.")
        return

    fig, ax = plt.subplots(figsize=(20, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(departed_evs)))

    for i, ev in enumerate(departed_evs):
        if not ev.historic_soc:
            continue

        start_time_h = ev.time_of_arrival * timescale / 60
        end_time_h = (ev.time_of_arrival + len(ev.historic_soc)) * timescale / 60
        time_vector_h = np.linspace(start_time_h, end_time_h, len(ev.historic_soc))

        ax.plot(time_vector_h, ev.historic_soc, color=colors[i], alpha=0.8, label=f'EV {ev.id}')

        desired_soc = ev.desired_capacity / ev.battery_capacity
        ax.hlines(y=desired_soc, xmin=start_time_h, xmax=ev.time_of_departure * timescale / 60,
                  color=colors[i], linestyle='--')

        departure_time_h = ev.time_of_departure * timescale / 60
        final_soc = ev.historic_soc[-1]
        ax.axvline(x=departure_time_h, color=colors[i], linestyle=':', linewidth=1)
        ax.plot(departure_time_h, final_soc, 'o', color=colors[i], markersize=8, markeredgecolor='black')

    ax.set_xlabel("Time (hours)", fontsize=14)
    ax.set_ylabel("State of Charge (SoC)", fontsize=14)
    ax.set_title(f"Individual Charge Profiles - Scenario: {scenario_name}\nAlgorithm: {algorithm_name}", fontsize=16)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(-0.05, 1.05)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # To avoid clutter, show legend only if there are not too many EVs
        if len(handles) < 20:
             ax.legend(handles, labels, title="EVs")

    plt.tight_layout()
    safe_algo_name = "".join(c for c in algorithm_name if c.isalnum() or c in ('_', '-')).rstrip()
    plot_filename = os.path.join(save_path, f"individual_ev_profiles_{scenario_name}_{safe_algo_name}.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Individual profiles plot saved to: {os.path.basename(plot_filename)}")

def plot_overload_composition(save_path, scenario_name, algorithm_name, power_data, timescale):
    ev_load = power_data.get('ev_load', [])
    inflexible_load = power_data.get('inflexible_load', [])
    solar_power = power_data.get('solar_power', [])
    limit = power_data.get('transformer_limit')
    if len(ev_load) == 0 or len(inflexible_load) == 0 or len(solar_power) == 0 or limit is None:
        print(f"Skipping overload plot for {algorithm_name} due to missing data.")
        return
    min_len = min(len(ev_load), len(inflexible_load), len(solar_power))
    time_hours = np.arange(min_len) * timescale / 60
    ev_load, inflexible_load, solar_power = ev_load[:min_len], inflexible_load[:min_len], solar_power[:min_len]
    net_load = ev_load + inflexible_load - solar_power
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.stackplot(time_hours, inflexible_load, ev_load, labels=['Inflexible Loads', 'EV Charging'], colors=['#3498db', '#9b59b6'], alpha=0.7)
    ax.fill_between(time_hours, 0, -solar_power, color='#2ecc71', alpha=0.6, label='Solar Generation')
    ax.plot(time_hours, net_load, label='Net Load', color='#e74c3c', linewidth=2.5)
    limit_series = np.full(min_len, limit) if np.isscalar(limit) else limit[:min_len]
    ax.plot(time_hours, limit_series, color='black', linestyle='--', linewidth=2, label=f'Transformer Limit')
    ax.fill_between(time_hours, net_load, limit_series, where=net_load > limit_series, facecolor='red', alpha=0.5, interpolate=True, label='Overload')
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Power (kW)")
    ax.set_title(f'Transformer Load Composition - {scenario_name} - Algorithm: {algorithm_name}', fontsize=16)
    ax.legend(loc='upper left'); ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=min(0, np.min(-solar_power)) * 1.1 if np.any(-solar_power) else 0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    safe_algo_name = "".join(c for c in algorithm_name if c.isalnum() or c in ('_', '-')).rstrip()
    plt.savefig(os.path.join(save_path, f"overload_composition_{scenario_name}_{safe_algo_name}.png"))
    plt.close(fig)

def plot_electricity_prices(save_path, scenario_name, charge_prices, discharge_prices, timescale):
    if charge_prices is None or discharge_prices is None: return
    charge_price_series, discharge_price_series = charge_prices[0], discharge_prices[0]
    min_len = min(len(charge_price_series), len(discharge_price_series))
    time_hours = np.arange(min_len) * timescale / 60
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(time_hours, charge_price_series[:min_len], label='Charge Price', color='#e74c3c', linewidth=2)
    ax.plot(time_hours, discharge_price_series[:min_len], label='Discharge Price', color='#2ecc71', linewidth=2, linestyle='--')
    ax.set_xlabel("Time (hours)"); ax.set_ylabel("Price (€/kWh)")
    ax.set_title(f'Electricity Prices - Scenario: {scenario_name}', fontsize=16)
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"electricity_prices_{scenario_name}.png"))
    plt.close(fig)

# =====================================================================================
# --- FUNZIONE PER GENERARE OUTPUT RIASSUNTIVI ---
# =====================================================================================
def generate_summary_outputs(stats_collection, save_path, scenario_name, num_simulations):
    column_mapping = {
        'total_profits': ('Profits/Costs (€)', 1), 'average_user_satisfaction': ('User Sat. (%)', 100),
        'total_energy_charged': ('Energy Ch. (kWh)', 1), 'total_energy_discharged': ('Energy Disch. (kWh)', 1),
        'transformer_overload_kwh': ('Tr. Ov. (kWh)', 1), 'battery_degradation': ('Total Qˡᵒˢᵗ (x10⁻³)', 1000),
        'execution_time': ('Exec. Time (s)', 1), 'total_reward': ('Reward (x10³)', 0.001)
    }
    summary_data = []
    for name, data in stats_collection.items():
        row = {'Algorithm': name}
        for key, (col_name, scale) in column_mapping.items():
            mean, std = data['mean'].get(key, 0) * scale, data['std'].get(key, 0) * scale
            row[col_name] = f"{mean:.2f} ± {std:.2f}" if num_simulations > 1 else f"{mean:.2f}"
        summary_data.append(row)
    if not summary_data: return
    df_summary = pd.DataFrame(summary_data).set_index('Algorithm')
    csv_path = os.path.join(save_path, f"summary_results_{scenario_name}.csv")
    df_summary.to_csv(csv_path)
    print(f"Aggregated results table saved to: {csv_path}")
    fig, ax = plt.subplots(figsize=(22, 1 + len(df_summary) * 0.5)); ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns, rowLabels=df_summary.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 1.3)
    for (row, col), cell in table.get_celld().items():
        if (row == 0 or col == -1): cell.set_text_props(weight='bold')
        if row > 0 and row % 2 == 1: cell.set_facecolor('#f2f2f2')
    fig.suptitle(f'Aggregated Results - Scenario: {scenario_name} (N={num_simulations} simulations)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    img_path = os.path.join(save_path, f"summary_table_{scenario_name}.png")
    plt.savefig(img_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"Aggregated results table image saved to: {img_path}")

# =====================================================================================
# --- FUNZIONE DI BENCHMARK ---
# =====================================================================================
def run_benchmark(config_files, reward_func, algorithms_to_run, num_simulations, model_dir, is_multi_scenario, price_data_file=None):
    overall_save_path = f'./results/benchmark_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(overall_save_path, exist_ok=True)

    max_obs_shape, max_action_shape = (0,), (0,)
    if is_multi_scenario:
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f: metadata = json.load(f)
            max_obs_shape, max_action_shape = tuple(metadata["observation_space_shape"]), tuple(metadata["action_space_shape"])
        else:
            temp_env = MultiScenarioEnv(config_files, reward_func, V2G_profit_max_loads)
            max_obs_shape, max_action_shape = temp_env.observation_space.shape, temp_env.action_space.shape
            temp_env.close()

    for config_file in config_files:
        scenario_name = os.path.basename(config_file).replace(".yaml", "")
        print(f"\n\n{'='*80}\nSTARTING BENCHMARK FOR SCENARIO: {scenario_name}\n{'='*80}")
        
        scenario_save_path = os.path.join(overall_save_path, scenario_name); os.makedirs(scenario_save_path, exist_ok=True)
        all_sim_stats = defaultdict(lambda: defaultdict(list))
        soc_over_time_by_algo = defaultdict(list)

        presence_env = EV2Gym(config_file=config_file, generate_rnd_game=True)
        print_desired_soc_summary(presence_env)
        ev_counts_over_time = []
        done = False
        while not done:
            ev_counts_over_time.append(sum(1 for cs in presence_env.charging_stations for ev in cs.evs_connected if ev is not None))
            _, _, done, _, _ = presence_env.step(np.zeros(presence_env.action_space.shape[0]))
        timescale, charge_prices, discharge_prices = presence_env.timescale, presence_env.charge_prices, presence_env.discharge_prices
        presence_env.close()

        for sim_num in range(num_simulations):
            print(f"\n--- Simulation {sim_num + 1}/{num_simulations} ---")
            env_replay = EV2Gym(config_file=config_file, generate_rnd_game=True, save_replay=True, price_data_file=price_data_file)
            replay_path = f"replay/replay_{env_replay.sim_name}.pkl"
            while not env_replay.step(np.zeros(env_replay.action_space.shape[0]))[2]: pass
            env_replay.close()

            eval_env_id = f'eval-env-{scenario_name}-{sim_num}'
            if eval_env_id in registry: del registry[eval_env_id]
            
            # =================================================================
            # --- KEY CHANGE: Activation of SoC recording ---
            # =================================================================
            gym.register(
                id=eval_env_id, 
                entry_point='ev2gym.models.ev2gym_env:EV2Gym', 
                kwargs={
                    'config_file': config_file, 
                    'generate_rnd_game': True, 
                    'load_from_replay_path': replay_path, 
                    'reward_function': reward_func, 
                    'state_function': V2G_profit_max_loads, 
                    'price_data_file': price_data_file,
                    'record_historic_soc': True  # <-- THIS IS THE ADDED LINE
                }
            )

            for name, (algorithm_class, rl_class, kwargs) in algorithms_to_run.items():
                print(f"+ Running: {name}")
                try:
                    env_instance = gym.make(eval_env_id)
                    is_rl_model = rl_class is not None
                    if is_rl_model:
                        if is_multi_scenario: env_instance = CompatibilityWrapper(env_instance, max_obs_shape, max_action_shape)
                        model_path = os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip')
                        if not os.path.exists(model_path): print(f"!!! Model {name} not found. Skipping."); continue
                        model = rl_class.load(model_path, env=env_instance, device="cuda" if torch.cuda.is_available() else "cpu")
                    else:
                        model = algorithm_class(env=env_instance.unwrapped, **kwargs)

                    start_time = time.time() # Start timing here
                    obs, _ = env_instance.reset()
                    done = False
                    current_run_soc_over_time = []
                    while not done:
                        action = model.predict(obs, deterministic=True)[0] if is_rl_model else model.get_action(env_instance.unwrapped)
                        obs, _, done, _, _ = env_instance.step(action)
                        connected_evs = [ev for cs in env_instance.unwrapped.charging_stations for ev in cs.evs_connected if ev is not None]
                        current_run_soc_over_time.append(np.mean([ev.get_soc() for ev in connected_evs]) if connected_evs else 0)
                    
                    soc_over_time_by_algo[name].append(current_run_soc_over_time)
                    
                    if sim_num == 0:
                        power_data = {
                            'ev_load': np.sum(env_instance.unwrapped.cs_power, axis=0),
                            'inflexible_load': np.sum(env_instance.unwrapped.tr_inflexible_loads, axis=0),
                            'solar_power': np.sum(env_instance.unwrapped.tr_solar_power, axis=0),
                            'transformer_limit': env_instance.unwrapped.transformers[0].max_power
                        }
                        plot_overload_composition(scenario_save_path, scenario_name, name, power_data, timescale)
                        
                        # Call to the new individual plot
                        plot_individual_ev_sessions(scenario_save_path, scenario_name, name, env_instance.unwrapped.departed_evs, timescale)

                    stats = env_instance.unwrapped.stats
                    stats['execution_time'] = time.time() - start_time # Calculate duration here
                    print(f"DEBUG: {name} execution_time: {stats['execution_time']:.4f} seconds") # Debug print
                    if departed_evs := env_instance.unwrapped.departed_evs:
                        stats['battery_degradation'] = np.mean([ev.get_battery_degradation() for ev in departed_evs])
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
            
            print("\n--- Generating output plots and summaries ---")
            plot_performance_metrics(aggregated_stats, scenario_save_path, scenario_name, list(algorithms_to_run.keys()))
            plot_ev_presence(scenario_save_path, scenario_name, ev_counts_over_time, timescale)
            plot_average_soc_over_time(scenario_save_path, scenario_name, soc_over_time_by_algo, timescale)
            plot_electricity_prices(scenario_save_path, scenario_name, charge_prices, discharge_prices, timescale)
            generate_summary_outputs(aggregated_stats, scenario_save_path, scenario_name, num_simulations)

    print(f"\n--- Benchmark finished. Results saved in: {overall_save_path} ---")

# =====================================================================================
# --- FUNZIONI DI CONFIGURAZIONE E ADDESTRAMENTO ---
# =====================================================================================
def calculate_max_cs(config_path: str) -> int:
    max_cs = 0
    for scenario_file in glob(os.path.join(config_path, "*.yaml")):
        with open(scenario_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            max_cs = max(max_cs, config.get('number_of_charging_stations', 0))
    if max_cs == 0: raise ValueError("Could not determine max number of charging stations.")
    return max_cs

def get_algorithms(max_cs: int, is_thesis_mode: bool) -> Dict[str, Tuple[Any, Any, Dict]]:
    base_algorithms = {
        "AFAP": (ChargeAsFastAsPossible, None, {}), 
        "ALAP": (ChargeAsLateAsPossible, None, {}), 
        "RR": (RoundRobin, None, {}),
        "SAC": (None, SAC, {}), 
        "DDPG+PER": (None, CustomDDPG, {'replay_buffer_class': PrioritizedReplayBuffer}),
        "TQC": (None, TQC, {}),
       
    }
    mpc_algorithms = {
        "Online_MPC_Profit_Max": (pulp_mpc.OnlineMPC_Solver, None, {'prediction_horizon': 25, 'control_horizon': 'half'}),
        "Online_MPC_Lyapunov_Adaptive": (pulp_mpc.OnlineMPC_Solver, None, {'prediction_horizon': 25, 'control_horizon': 'half', 'lyapunov_adaptive': True}) # Placeholder for Lyapunov adaptive kwargs
    }
    ALL_ALGORITHMS = {**base_algorithms, **mpc_algorithms}
    THESIS_ALGORITHMS_BASE = ["AFAP", "ALAP", "RR", "SAC", "DDPG", "DDPG+PER", "TQC", "Online_MPC_Profit_Max", "Online_MPC_Lyapunov_Adaptive"]
    THESIS_ALGORITHMS = {k: v for k, v in ALL_ALGORITHMS.items() if k in THESIS_ALGORITHMS_BASE}
    return THESIS_ALGORITHMS if is_thesis_mode else ALL_ALGORITHMS

def train_rl_models_if_requested(scenarios_to_test: List[str], selected_reward_func: Callable, algorithms_to_run: Dict, is_multi_scenario: bool, model_dir: str, selected_price_file_abs_path: Optional[str], steps_for_training: int, training_mode: str = 'single', curriculum_steps_per_level: int = 10000) -> None:
    rl_models_to_run = {k: v for k, v in algorithms_to_run.items() if v[1] is not None}
    if not rl_models_to_run: return
    
    if training_mode == 'single':
        train_env = Monitor(gym.make('ev2gym.models.ev2gym_env:EV2Gym', config_file=scenarios_to_test[0], generate_rnd_game=True, reward_function=selected_reward_func, state_function=V2G_profit_max_loads, price_data_file=selected_price_file_abs_path))
    else:
        env_map = {'random': MultiScenarioEnv, 'curriculum': CurriculumEnv, 'shuffled': ShuffledMultiScenarioEnv}
        env_class = env_map.get(training_mode, MultiScenarioEnv)
        kwargs = {'steps_per_level': curriculum_steps_per_level} if training_mode == 'curriculum' else {}
        train_env = DummyVecEnv([lambda: Monitor(env_class(scenarios_to_test, selected_reward_func, V2G_profit_max_loads, **kwargs))])

    for name, (_, rl_class, kwargs) in rl_models_to_run.items():
        print(f"--- Training {name} ---")
        model = rl_class("MlpPolicy", train_env, verbose=0, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs)
        model.learn(total_timesteps=steps_for_training, callback=[ProgressCallback(steps_for_training), TrainingPlotCallback(name)])
        model.save(os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip'))

    if is_multi_scenario:
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump({"observation_space_shape": list(train_env.observation_space.shape), "action_space_shape": list(train_env.action_space.shape)}, f, indent=4)
    train_env.close()
