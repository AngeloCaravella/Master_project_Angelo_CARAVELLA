import streamlit as st
import os
import sys
import subprocess
from glob import glob
import inspect
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

# Aggiungi la directory del progetto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# RL library imports for training
from stable_baselines3 import SAC, DDPG, TD3, PPO
from sb3_contrib import TQC

# Import functions from run_experiments
from run_experiments import (
    calculate_max_cs,
    get_algorithms,
    train_rl_models_if_requested,
    run_benchmark
)
from ev2gym.rl_agent import reward as reward_module

st.set_page_config(layout="wide", page_title="EV2Gym Dashboard")

st.title("EV2Gym Dashboard")
st.markdown("Visualize simulation results, analyze configurations, and calibrate the battery model.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: **Angelo Caravella**")

# --- Helper functions ---

def get_available_rl_algorithms():
    """
    Returns a dictionary of RL algorithms compatible with continuous action spaces.
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

# Streamlit-compatible input functions
def st_get_interactive_input(prompt, default=None, key=None):
    return st.text_input(prompt, value=str(default), key=key)

def st_select_from_list(items, prompt, multiple=False, default_choice=None, key=None):
    options = [os.path.basename(item).replace('.yaml', '') if isinstance(item, str) and item.endswith('.yaml') else item for item in items]
    
    if multiple:
        default_values = []
        if default_choice is not None:
            if isinstance(default_choice, list):
                default_values = [options[i-1] for i in default_choice if 0 < i <= len(options)]
            elif isinstance(default_choice, int) and 0 < default_choice <= len(options):
                default_values = [options[default_choice-1]]
        
        selected_options = st.multiselect(prompt, options=options, default=default_values, key=key)
        return [items[options.index(opt)] for opt in selected_options]
    else:
        default_index = 0
        if default_choice is not None:
            if isinstance(default_choice, int) and 0 < default_choice <= len(options):
                default_index = default_choice - 1
            elif isinstance(default_choice, str) and default_choice in options:
                default_index = options.index(default_choice)
        
        selected_option = st.selectbox(prompt, options=options, index=default_index, key=key)
        return items[options.index(selected_option)]

# --- Common paths and data ---
config_path = os.path.join(project_root, "ev2gym", "example_config_files")
available_scenarios_full_paths = sorted(glob(os.path.join(config_path, "*.yaml")))
price_data_dir = os.path.join(project_root, 'ev2gym', 'data')
available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
default_price_file = "Netherlands_day-ahead-2015-2024.csv"
try:
    default_price_index = available_price_files.index(default_price_file) + 1
except ValueError:
    default_price_index = 1

# --- Run Fit_battery.py ---
if st.sidebar.button("Calibrate Battery Model (Fit_battery.py)"):
    st.subheader("Executing Fit_battery.py")
    with st.spinner("Calibration in progress..."):
        try:
            process = subprocess.run(["python", os.path.join(project_root, "Fit_battery.py")], capture_output=True, text=True, check=True)
            st.success("Calibration completed successfully!")
            st.code(process.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"Error during execution of Fit_battery.py (code {e.returncode}):")
            st.code(e.stderr)
        except FileNotFoundError:
            st.error("Error: The script 'Fit_battery.py' was not found.")

# --- NEW SECTION: Train RL Models ---
st.subheader("Train New RL Models")

with st.expander("Training Configuration"):
    # 1. Select algorithms to train
    available_rl_algos = get_available_rl_algorithms()
    selected_algo_names = st.multiselect(
        "Select the RL algorithms to train:",
        options=list(available_rl_algos.keys()),
        default=[list(available_rl_algos.keys())[0]] if available_rl_algos else [],
        key="train_algos_select"
    )

    # 2. Training configuration
    st.info("Training is performed in 'Dynamic Parameters' mode, which trains models with dynamically varied parameters for greater robustness.")
    base_scenario_path = st_select_from_list(available_scenarios_full_paths, "Select a BASE scenario for parameter randomization:", key="train_base_scenario")

    steps_for_training = st.number_input("Total training steps:", min_value=1000, value=100000, step=1000, key="train_steps")
    
    default_session_name = f"dynamic_{'_'.join(selected_algo_names).lower()}_{datetime.datetime.now().strftime('%Y%m%d')}"
    session_name = st.text_input("Enter a name for this training session:", value=default_session_name, key="train_session_name")

    # 3. Select reward and price file
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    reward_names = [name for name, _ in available_rewards]
    selected_reward_name_train = st_select_from_list(reward_names, "Select reward function for training:", default_choice=1, key="train_reward_func_select")
    selected_reward_func_train = next(func for name, func in available_rewards if name == selected_reward_name_train)

    selected_price_file_name_train = st_select_from_list(available_price_files, "Select CSV file for energy prices for training:", default_choice=default_price_index, key="train_price_file_select")
    selected_price_file_abs_path_train = os.path.join(price_data_dir, selected_price_file_name_train)

    if st.button("Start Training", key="start_training_button"):
        if not selected_algo_names:
            st.error("Please select at least one RL algorithm to train.")
        elif not base_scenario_path:
            st.error("Please select a base scenario.")
        else:
            algorithms_to_train = {k: available_rl_algos[k] for k in selected_algo_names}
            model_dir_train = os.path.join(project_root, 'saved_models', "".join(c for c in session_name if c.isalnum() or c in ("_", "-")).rstrip(), '')
            
            st.markdown("---")
            st.subheader("Training RL Models")
            with st.spinner(f"Training in progress... Models will be saved to: {model_dir_train}"):
                try:
                    os.makedirs(model_dir_train, exist_ok=True)

                    train_rl_models_if_requested(
                        scenarios_to_test=[base_scenario_path],
                        selected_reward_func=selected_reward_func_train,
                        algorithms_to_run=algorithms_to_train,
                        is_multi_scenario=True,
                        model_dir=model_dir_train,
                        selected_price_file_abs_path=selected_price_file_abs_path_train,
                        steps_for_training=steps_for_training,
                        training_mode='dynamic',
                        session_name=session_name
                    )
                    st.success("Training completed successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
                    st.exception(e)

# --- NEW SECTION: Run Simulations (Benchmark) ---
st.subheader("Run Benchmark and Plot Results")

with st.expander("Benchmark Configuration", expanded=True):
    # 1. Select the models folder
    saved_models_dir = os.path.join(project_root, 'saved_models')
    if not os.path.exists(saved_models_dir) or not os.listdir(saved_models_dir):
        st.warning(f"No model folders found in '{saved_models_dir}'. You can only run baseline algorithms.")
        available_models = []
    else:
        available_models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])

    model_dir = None
    is_multi_scenario = False
    trained_rl_algos = []

    if not available_models:
        st.info("No trained models found. Only baseline algorithms are available.")
        selected_model_name = None
    else:
        selected_model_name = st_select_from_list(available_models, "Select the set of models to load for benchmarking:", key="benchmark_model_set_select")

    if selected_model_name:
        model_dir = os.path.join(saved_models_dir, selected_model_name)
        is_multi_scenario = True # Per run_interactive.py, this is the default for plotting
        st.info(f"Models selected from: {model_dir} (multi-scenario mode by default)")

        # 2. Detect available algorithms in that folder
        available_model_files = glob(os.path.join(model_dir, '*_model.zip'))
        trained_rl_algos = [os.path.basename(f).replace('_model.zip', '').replace('_', '+').upper() for f in available_model_files]
        st.write(f"Trained RL algorithms found in this session: {trained_rl_algos}")

    # 3. Get base algorithms (Heuristics, MPC) and RL definitions
    MAX_CS = calculate_max_cs(config_path)
    st.info(f"Detected a maximum of {MAX_CS} charging stations across all scenarios.")
    
    # In run_interactive, is_thesis_mode is True for plotting. We'll stick to that for simplicity and correctness.
    all_base_algos = get_algorithms(MAX_CS, is_thesis_mode=True)
    baselines = {k: v for k, v in all_base_algos.items() if v[1] is None}
    
    all_rl_definitions = get_available_rl_algorithms()
    available_rl_from_files = {}
    for rl_name in trained_rl_algos: # e.g., 'DDPG+PER'
        base_name = rl_name.split('+')[0] # e.g., 'DDPG'
        if base_name in all_rl_definitions:
            available_rl_from_files[rl_name] = all_rl_definitions[base_name]

    all_available_definitions = {**baselines, **available_rl_from_files}

    # 4. Select algorithms to plot
    plot_candidates = sorted(list(baselines.keys())) + sorted(list(available_rl_from_files.keys()))
    
    if not plot_candidates:
        st.error("No algorithms available for benchmarking.")
        selected_for_plot = []
    else:
        selected_for_plot = st.multiselect(
            "Select the algorithms to compare in the benchmark:",
            options=plot_candidates,
            default=plot_candidates,
            key="benchmark_algos_select"
        )

    algorithms_to_run = {k: all_available_definitions[k] for k in selected_for_plot if k in all_available_definitions}
    
    # --- Specific configuration for OnlineMPC_Solver ---
    online_mpc_keys = [k for k in algorithms_to_run if 'Online_MPC' in k]
    if online_mpc_keys:
        st.markdown("---")
        st.markdown("### Online MPC Configuration (for control horizon > 1)")
        pred_h = st.number_input("Enter prediction horizon (Np):", min_value=1, value=5, key="benchmark_pred_h_input")
        ctrl_h_input = st.text_input("Enter control horizon (Nc) (e.g., 1, 3, or 'half'):", value="1", key="benchmark_ctrl_h_input")

        try:
            ctrl_h = int(ctrl_h_input)
        except ValueError:
            ctrl_h = ctrl_h_input

        for key in online_mpc_keys:
            if key in algorithms_to_run:
                mpc_kwargs = algorithms_to_run[key][2]
                mpc_kwargs['prediction_horizon'] = pred_h
                mpc_kwargs['control_horizon'] = ctrl_h
                st.write(f"Online MPC '{key}' updated: Np={pred_h}, Nc={ctrl_h}")

    # 5. Benchmark configuration
    scenarios_to_test = st_select_from_list(available_scenarios_full_paths, "Select the scenarios for the BENCHMARK:", multiple=True, key="benchmark_scenarios_select")
    
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    reward_names = [name for name, _ in available_rewards]
    selected_reward_name = st_select_from_list(reward_names, "Choose the reward function (ensure it matches training):", default_choice=1, key="benchmark_reward_func_select")
    selected_reward_func = next(func for name, func in available_rewards if name == selected_reward_name)

    selected_price_file_name = st_select_from_list(available_price_files, "Select the CSV file for energy prices:", default_choice=default_price_index, key="benchmark_price_file_select")
    selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)

    num_sims = st.number_input("How many evaluation simulations per scenario?", min_value=1, value=1, key="benchmark_num_sims_input")

    # 6. Execute benchmark
    if st.button("Run Benchmark Simulations", key="run_benchmark_button"):
        if not scenarios_to_test:
            st.error("Please select at least one scenario for the benchmark.")
        elif not algorithms_to_run:
            st.error("Please select at least one algorithm to run.")
        elif model_dir is None and any(v[1] is not None for v in algorithms_to_run.values()):
            st.error("An RL algorithm was selected, but no model set was loaded. Please select a model set.")
        else:
            st.markdown("---")
            st.subheader("Benchmark Results")
            with st.spinner("Running benchmark simulations... This may take a while."):
                try:
                    run_benchmark(
                        config_files=scenarios_to_test,
                        reward_func=selected_reward_func,
                        algorithms_to_run=algorithms_to_run,
                        num_simulations=num_sims,
                        model_dir=model_dir,
                        is_multi_scenario=is_multi_scenario,
                        price_data_file=selected_price_file_abs_path
                    )
                    st.success("Benchmark simulations completed successfully!")
                    st.balloons()

                    # --- Display Results from the latest run ---
                    results_base_path = os.path.join(project_root, "results")
                    benchmark_folders = sorted([f.name for f in os.scandir(results_base_path) if f.is_dir()], reverse=True)
                    if benchmark_folders:
                        latest_benchmark_folder = benchmark_folders[0]
                        selected_folder_path = os.path.join(results_base_path, latest_benchmark_folder)
                        st.markdown(f"### Results from latest run: {latest_benchmark_folder}")

                        # Display images
                        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                        all_image_paths = []
                        for root, _, files in os.walk(selected_folder_path):
                            for file in files:
                                if file.lower().endswith(image_extensions):
                                    all_image_paths.append(os.path.join(root, file))
                        all_image_paths.sort()

                        if all_image_paths:
                            for img_path in all_image_paths:
                                relative_path = os.path.relpath(img_path, selected_folder_path)
                                st.image(img_path, caption=relative_path, use_container_width=True)
                        else:
                            st.info("No images found for the latest benchmark run.")

                        # Display CSV summaries
                        csv_paths = sorted(glob(os.path.join(selected_folder_path, "**", "summary_results_*.csv"), recursive=True))
                        if csv_paths:
                            st.markdown("### Summary Tables (CSV)")
                            for csv_path in csv_paths:
                                st.markdown(f"**{os.path.basename(csv_path)}**")
                                df = pd.read_csv(csv_path)
                                st.dataframe(df)
                        else:
                            st.info("No summary CSVs found for the latest benchmark run.")

                    else:
                        st.info("No benchmark results found to display.")

                except Exception as e:
                    st.error(f"An error occurred during benchmark execution: {e}")
                    st.exception(e)

# --- Existing section for ANALISI CONFIGURAZIONI (now translated) ---
available_scenario_names = [os.path.basename(s).replace(".yaml", "") for s in available_scenarios_full_paths]

with st.expander("Scenario Configuration Analysis (.yaml)"):
    st.markdown("View summary tables of parameters for selected scenarios.")
    
    selected_scenario_names = st.multiselect(
        "Select scenarios to analyze:",
        options=available_scenario_names,
        default=[]
    )
    scenarios_to_analyze = [s for s in available_scenarios_full_paths if os.path.basename(s).replace(".yaml", "") in selected_scenario_names]

    if st.button("Generate and View Summary Tables"):
        if not scenarios_to_analyze:
            st.warning("Please select at least one scenario from the list.")
        else:
            with st.spinner("Analyzing configuration files..."):
                try:
                    from Compare import generate_summary_figures_for_streamlit
                    
                    summary_figures = generate_summary_figures_for_streamlit(config_path, scenarios_to_analyze)
                    
                    if not summary_figures:
                        st.error("Unable to generate tables. No valid data extracted from files.")
                    else:
                        st.success(f"Tables generated successfully for {len(scenarios_to_analyze)} scenarios.")
                        for fig in summary_figures:
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"An error occurred during table generation: {e}")


# =============================================================================
# --- Visualizzatore Risultati (now dynamic) ---
# =============================================================================
st.subheader("Results Viewer")

results_base_path = os.path.join(project_root, "results")

if not os.path.exists(results_base_path):
    st.warning(f"The results folder '{results_base_path}' does not exist yet.")
else:
    benchmark_folders = [f.name for f in os.scandir(results_base_path) if f.is_dir()]
    benchmark_folders.sort(reverse=True)

    if not benchmark_folders:
        st.info("No benchmark folders found in the results directory.")
    else:
        selected_benchmark_folder = st.selectbox(
            "Select a benchmark folder:",
            options=benchmark_folders,
            key="results_viewer_folder_select"
        )

        if selected_benchmark_folder:
            selected_folder_path = os.path.join(results_base_path, selected_benchmark_folder)
            
            sub_folders = sorted([f.name for f in os.scandir(selected_folder_path) if f.is_dir()])
            
            sub_folder_options = ["All subfolders"] + sub_folders
            
            selected_sub_folder = st.selectbox(
                "Select a subfolder (or all):",
                options=sub_folder_options,
                key="results_viewer_subfolder_select"
            )

            if selected_sub_folder == "All subfolders":
                search_path = selected_folder_path
                display_caption_base = selected_folder_path
            else:
                search_path = os.path.join(selected_folder_path, selected_sub_folder)
                display_caption_base = search_path

            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
            all_image_paths = []
            for root, _, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        all_image_paths.append(os.path.join(root, file))
            all_image_paths.sort()

            if not all_image_paths:
                st.info(f"No images found in '{selected_sub_folder}'.")
            else:
                for img_path in all_image_paths:
                    relative_path = os.path.relpath(img_path, display_caption_base)
                    st.image(img_path, caption=relative_path, use_container_width=True)
            
            # Display CSV summaries in the results viewer
            csv_paths = sorted(glob(os.path.join(search_path, "**", "summary_results_*.csv"), recursive=True))
            if csv_paths:
                st.markdown("### Summary Tables (CSV)")
                for csv_path in csv_paths:
                    st.markdown(f"**{os.path.basename(csv_path)}**")
                    df = pd.read_csv(csv_path)
                    st.dataframe(df)
            else:
                st.info("No summary CSVs found in the selected folder.")
