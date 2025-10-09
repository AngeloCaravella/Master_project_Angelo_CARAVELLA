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

# Import functions from run_experiments
from run_experiments import (
    calculate_max_cs,
    get_algorithms,
    run_benchmark
)
from ev2gym.rl_agent import reward as reward_module

st.set_page_config(layout="wide", page_title="EV2Gym Dashboard")

st.title("EV2Gym Dashboard")
st.markdown("Visualize simulation results, analyze configurations, and calibrate the battery model.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: **Angelo Caravella**")

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

# --- NEW SECTION: Run Simulations ---
st.subheader("Run Simulations")

with st.expander("Simulation Configuration"):
    # --- Plot Mode ---
    plot_mode_choice = st.radio("Select plot mode:", ["Thesis (1)", "Complete (2)"], index=0, key="plot_mode_radio")
    is_thesis_mode = (plot_mode_choice == 'Thesis (1)')

    # --- Calculate MAX_CS ---
    config_path = os.path.join(project_root, "ev2gym", "example_config_files")
    MAX_CS = calculate_max_cs(config_path)
    st.info(f"Detected a maximum of {MAX_CS} charging stations across all scenarios.")

    # --- MPC Type ---
    mpc_type_choice = st.radio("Select MPC type:", ["linear", "quadratic"], index=0, key="mpc_type_radio")

    # --- Get Algorithms ---
    algorithms_to_run = get_algorithms(MAX_CS, is_thesis_mode, mpc_type_choice)
    st.write(f"Algorithms to be executed: {list(algorithms_to_run.keys())}")

    # --- Specific configuration for OnlineMPC_Solver ---
    online_mpc_keys = [k for k in algorithms_to_run if 'Online_MPC' in k]
    if online_mpc_keys:
        st.markdown("---")
        st.markdown("### Online MPC Configuration (for control horizon > 1)")
        pred_h = st.number_input("Enter prediction horizon (Np):", min_value=1, value=5, key="pred_h_input")
        ctrl_h_input = st.text_input("Enter control horizon (Nc) (e.g., 1, 3, or 'half'):", value="1", key="ctrl_h_input")

        try:
            ctrl_h = int(ctrl_h_input)
        except ValueError:
            ctrl_h = ctrl_h_input

        for key in online_mpc_keys:
            mpc_kwargs = algorithms_to_run[key][2]
            mpc_kwargs['prediction_horizon'] = pred_h
            mpc_kwargs['control_horizon'] = ctrl_h
            st.write(f"Online MPC '{key}' updated: Np={pred_h}, Nc={ctrl_h}")

    # --- Select Scenarios ---
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    scenarios_to_test = st_select_from_list(available_scenarios, "Select scenarios to test:", multiple=True, key="scenarios_to_test_select")
    st.write(f"Selected scenarios: {[os.path.basename(s) for s in scenarios_to_test]}")

    # --- Select Reward Function ---
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    reward_names = [name for name, _ in available_rewards]
    selected_reward_name = st_select_from_list(reward_names, "Select reward function:", default_choice=1, key="reward_func_select")
    selected_reward_func = next(func for name, func in available_rewards if name == selected_reward_name)
    st.write(f"Selected reward function: {selected_reward_name}")

    # --- Select Price File ---
    price_data_dir = os.path.join(project_root, 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    try:
        default_price_index = available_price_files.index(default_price_file) + 1
    except ValueError:
        default_price_index = 1
    
    selected_price_file_name = st_select_from_list(available_price_files, "Select CSV file for energy prices:", default_choice=default_price_index, key="price_file_select")
    selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)
    st.write(f"Selected price file: {selected_price_file_name}")

    # --- Select Pre-trained Model ---
    saved_models_dir = os.path.join(project_root, 'saved_models')
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
    
    available_models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
    
    model_dir = None
    is_multi_scenario = False

    if not available_models:
        st.warning("No trained models found in './saved_models/'. Please train models using run_interactive.py first if you want to use RL algorithms.")
    else:
        selected_model_name = st_select_from_list(available_models, "Select the set of models to load:", key="model_set_select")
        model_dir = os.path.join(saved_models_dir, selected_model_name)
        
        if 'multi' in selected_model_name.lower() or 'curriculum' in selected_model_name.lower():
            is_multi_scenario = True
        else:
            is_multi_scenario = False
        
        st.info(f"Models selected from: {model_dir}")
        st.info(f"Detected mode: {'multi-scenario' if is_multi_scenario else 'single-scenario'}")
        st.warning("WARNING: Ensure that the selected test scenarios are compatible with the loaded models.")

    # --- Number of Simulations ---
    num_sims = st.number_input("Number of evaluation simulations per scenario:", min_value=1, value=1, key="num_sims_input")

    if st.button("Run Benchmark Simulations", key="run_benchmark_button"):
        if not scenarios_to_test:
            st.error("Please select at least one scenario to run simulations.")
        elif model_dir is None and any(v[1] is not None for v in algorithms_to_run.values()):
            st.error("Please select a trained model set to run RL algorithms.")
        else:
            st.markdown("---")
            st.subheader("Simulation Results")
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