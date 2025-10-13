# --- START OF FILE run_interactive.py ---

import os
import inspect
from glob import glob
import subprocess
import time
import yaml
import itertools
import numpy as np
from pyDOE import lhs
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

# Import functions from run_experiments
from run_experiments import (
    calculate_max_cs,
    get_algorithms,
    train_rl_models_if_requested,
    run_benchmark
)
from ev2gym.rl_agent import reward as reward_module

def set_nested_dict_value(d, path, value):
    """Sets a value in a nested dictionary using a list of keys."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value

def get_interactive_input(prompt, default=None):
    """Helper function to get user input with a default value."""
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input or default

def select_from_list(items, prompt, multiple=False, default_choice=1):
    """Helper function to let the user select one or more items from a list."""
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        # Handle both strings and tuples (like reward functions)
        display_name = item if isinstance(item, str) else item[0]
        print(f"{i+1}. {os.path.basename(display_name).replace('.yaml', '')}")
    
    if multiple:
        choices = input(f"Seleziona uno o più (es. '1 3', 'tutti') (default: tutti): ").lower() or 'tutti'
        if 'tutti' in choices:
            return items
        else:
            try:
                return [items[int(i)-1] for i in choices.split()]
            except (ValueError, IndexError):
                print("Selezione non valida. Verranno usati tutti gli elementi.")
                return items
    else:
        try:
            choice = input(f"Scelta (default: {default_choice}): ") or str(default_choice)
            return items[int(choice)-1]
        except (ValueError, IndexError):
            print(f"Selezione non valida. Verrà usata la scelta di default ({default_choice}).")
            return items[default_choice-1]

def main():
    """Main function to run the interactive simulation."""

    # --- Run Fit_battery.py ---
    if get_interactive_input("Vuoi eseguire 'Fit_battery.py' per calibrare il modello di degradazione? (s/n)", "n").lower() == 's':
        print("--- Esecuzione di Fit_battery.py ---")
        try:
            subprocess.run(["python", "Fit_battery.py"], check=True)
            print("--- Fit_battery.py completato. ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERRORE: {e}. Lo script procederà con i parametri esistenti.")

    # --- Plot Mode ---
    plot_mode_choice = get_interactive_input("Scegli modalità grafici: '1' per Tesi (consigliato), '2' per Completa", "1")
    is_thesis_mode = (plot_mode_choice == '1')

    # --- Calculate MAX_CS ---
    config_path = "ev2gym/example_config_files/"
    detected_max_cs = calculate_max_cs(config_path)
    print(f"\nRilevato un massimo di {detected_max_cs} stazioni di ricarica tra tutti gli scenari.")
    
    user_max_cs = get_interactive_input(f"Inserisci il numero massimo di stazioni di ricarica per cui addestrare gli algoritmi RL (default: {detected_max_cs})", str(detected_max_cs))
    MAX_CS = int(user_max_cs)

    # --- Get ALL available algorithms ---
    all_available_algorithms = get_algorithms(MAX_CS, is_thesis_mode)
    
    # --- Interactive Algorithm Selection ---
    # Identify all "advanced" controllers (MPC and Optimal) for user selection
    advanced_solver_keys = [k for k in all_available_algorithms if "MPC" in k or "Optimal" in k]
    
    # Other algorithms (heuristics and RL) are kept separate
    base_algorithms = {k: v for k, v in all_available_algorithms.items() if k not in advanced_solver_keys}

    if advanced_solver_keys:
        selected_advanced_keys = select_from_list(
            advanced_solver_keys, 
            "Seleziona quali controller avanzati (MPC, Optimal) eseguire:", 
            multiple=True
        )
        selected_advanced = {k: all_available_algorithms[k] for k in selected_advanced_keys}
        # The final list for the benchmark will include both base and selected advanced algorithms
        algorithms_to_run = {**base_algorithms, **selected_advanced}
    else:
        algorithms_to_run = base_algorithms

    print(f"\nAlgoritmi che verranno eseguiti nel benchmark: {list(algorithms_to_run.keys())}")

    # --- Algorithm-specific Configurations ---
    if 'DDPG+PER' in algorithms_to_run:
        print("\n--- Configurazione Prioritized Experience Replay (PER) per DDPG+PER ---")
        per_alpha = float(get_interactive_input("Inserisci il valore di alpha (livello di priorità)", "0.6"))
        per_beta = float(get_interactive_input("Inserisci il valore iniziale di beta (correzione bias)", "0.4"))
        ddpg_per_kwargs = algorithms_to_run['DDPG+PER'][2]
        ddpg_per_kwargs['replay_buffer_kwargs'] = {'alpha': per_alpha, 'beta': per_beta}



    # --- Select Scenarios for Benchmark ---
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    benchmark_scenarios = select_from_list(available_scenarios, "Seleziona gli scenari per il BENCHMARK:", multiple=True)
    print(f"Scenari di benchmark selezionati: {[os.path.basename(s) for s in benchmark_scenarios]}")

    # --- Select Reward Function ---
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    selected_reward_tuple = select_from_list(available_rewards, "Scegli la funzione di reward:", default_choice=1)
    selected_reward_func = selected_reward_tuple[1]
    print(f"Funzione di reward selezionata: {selected_reward_tuple[0]}")

    # --- Select Price File ---
    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    try:
        default_price_index = available_price_files.index(default_price_file) + 1
    except ValueError:
        default_price_index = 1
    selected_price_file_name = select_from_list(available_price_files, "Seleziona il file CSV per i prezzi dell'energia:", default_choice=default_price_index)
    selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)
    print(f"File prezzi selezionato: {os.path.basename(selected_price_file_name)}")

    # --- Train or Load RL Models ---
    train_rl_models = get_interactive_input("Vuoi addestrare i modelli RL? (s/n)", "n").lower() == 's'
    model_dir = ''
    is_multi_scenario = False

    if train_rl_models:
        # Create a filtered list of algorithms for training (excluding non-trainable solvers)
        algorithms_for_training = algorithms_to_run.copy()
        non_trainable_solvers = [k for k in algorithms_for_training if "Optimal" in k]
        if non_trainable_solvers:
            print("\nNOTA: I seguenti solver di benchmark verranno esclusi dall'addestramento:")
            for solver_key in non_trainable_solvers:
                print(f" - {solver_key}")
                del algorithms_for_training[solver_key]
        
        prompt = ("\nScegli la modalità di addestramento:\n" 
                  "  '1' per Scenario Singolo\n"
                  "  '2' per Multi-Scenario Casuale (con reinserimento)\n"
                  "  '3' per Curriculum Learning\n"
                  "  '4' per Casuale a Epoche (senza reinserimento)\n"
                  "  '5' per Addestramento a Parametri Dinamici")
        mode_choice = get_interactive_input(prompt, "5")
        
        training_scenarios = []
        training_mode = 'random'
        curriculum_steps_per_level = 10000

        if mode_choice == '1':
            training_mode = 'single'
            training_scenarios = [select_from_list(available_scenarios, "Seleziona lo scenario per l'addestramento:")]
            is_multi_scenario = False
        elif mode_choice == '5':
            training_mode = 'dynamic'
            is_multi_scenario = True
            print("\n--- Configurazione Addestramento a Parametri Dinamici ---")
            print(f"NOTA: Questa modalità addestrerà i modelli con parametri variati dinamicamente, incluso il numero di stazioni di ricarica fino a 100 CS (valore massimo intrinseco). Si consiglia di impostare il MAX_CS a 100 per coerenza.")
            base_scenario_path = select_from_list(available_scenarios, "Seleziona uno scenario di BASE per la randomizzazione dei parametri:")
            training_scenarios = [base_scenario_path]
        else: # All other modes are multi-scenario
            is_multi_scenario = True
            if mode_choice == '2':
                training_mode = 'random'
                training_scenarios = select_from_list(available_scenarios, "Seleziona gli scenari per l'addestramento multi-scenario:", multiple=True)
            elif mode_choice == '3':
                training_mode = 'curriculum'
                print("\n--- Configurazione Curriculum Learning ---")
                order_input = input(f"Specifica l'ordine degli scenari (es. '1 3 2' da {len(available_scenarios)} disponibili): ")
                order = [int(i)-1 for i in order_input.split()]
                training_scenarios = [available_scenarios[i] for i in order]
                print(f"Curriculum definito: {[os.path.basename(s) for s in training_scenarios]}")
                curriculum_steps_per_level = int(get_interactive_input("Passi di training per livello?", "10000"))
            elif mode_choice == '4':
                training_mode = 'shuffled'
                training_scenarios = select_from_list(available_scenarios, "Seleziona gli scenari per l'addestramento Casuale a Epoche:", multiple=True)

        steps_for_training = int(get_interactive_input("Per quanti passi di training totali?", "100000"))
        
        session_name = get_interactive_input("Inserisci un nome per questa sessione di addestramento", f"{training_mode}_{time.strftime('%Y%m%d')}")
        model_dir = f'./saved_models/{"".join(c for c in session_name if c.isalnum() or c in ("_", "-")).rstrip()}/'
        os.makedirs(model_dir, exist_ok=True)

        train_rl_models_if_requested(
            scenarios_to_test=training_scenarios,
            selected_reward_func=selected_reward_func,
            algorithms_to_run=algorithms_for_training,  # <-- Use the filtered list
            is_multi_scenario=is_multi_scenario,
            model_dir=model_dir,
            selected_price_file_abs_path=selected_price_file_abs_path,
            steps_for_training=steps_for_training,
            training_mode=training_mode,
            curriculum_steps_per_level=curriculum_steps_per_level,
            session_name=session_name
        )
    else:
        saved_models_dir = './saved_models/'
        if not os.path.exists(saved_models_dir) or not os.listdir(saved_models_dir):
            print("\nERRORE: Nessun modello addestrato trovato in './saved_models/'. Esegui prima l'addestramento.")
            return
        
        available_models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
        selected_model_name = select_from_list(available_models, "Seleziona il set di modelli da caricare:", multiple=False)
        model_dir = os.path.join(saved_models_dir, selected_model_name)
        
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            is_multi_scenario = True
            print(f"Rilevato file di metadati: i modelli sono multi-scenario.")
        else:
            is_multi_scenario = any(keyword in selected_model_name.lower() for keyword in ['multi', 'curriculum', 'shuffled', 'random'])
            print(f"Nessun file di metadati. Modalità rilevata dal nome: {'multi-scenario' if is_multi_scenario else 'scenario singolo'}")
        
        print(f"\nModelli selezionati da: {model_dir}")
        print("ATTENZIONE: Assicurati che gli scenari di benchmark siano compatibili con i modelli caricati.")

        # --- Read and display neural network dimensions ---
        print("\n--- Dimensioni delle reti neurali dei modelli RL caricati ---")
        
        # Get the list of RL algorithms from all_available_algorithms
        rl_algorithms_info = {k: v for k, v in all_available_algorithms.items() if v[1] is not None}

        if not rl_algorithms_info:
            print("Nessun algoritmo RL configurato per l'ispezione.")
        else:
            # Determine target shapes from metadata if available (for multi-scenario models)
            target_obs_shape = None
            target_action_shape = None
            if is_multi_scenario and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    target_obs_shape = tuple(metadata["observation_space_shape"])
                    target_action_shape = tuple(metadata["action_space_shape"])
                    print(f"Dimensioni target dell'ambiente (da metadati): Osservazione={target_obs_shape}, Azione={target_action_shape}")
                except Exception as e:
                    print(f"ERRORE durante la lettura dei metadati del modello: {e}")
                    target_obs_shape = None
                    target_action_shape = None
            elif not is_multi_scenario:
                print("NOTA: Per i modelli addestrati su scenario singolo, le dimensioni della rete potrebbero non essere accurate senza l'ambiente di addestramento originale.")

            for algo_name, (algo_class, rl_class, kwargs) in rl_algorithms_info.items():
                model_file_name = f'{algo_name.lower().replace("+", "_")}_model.zip'
                model_path = os.path.join(model_dir, model_file_name)

                if os.path.exists(model_path):
                    try:
                        # Create a dummy environment if target shapes are known
                        temp_env_for_load = None
                        if target_obs_shape and target_action_shape:
                            dummy_env_instance = gym.make("CartPole-v1") # Placeholder, will be replaced
                            dummy_env_instance.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=target_obs_shape, dtype=np.float64)
                            dummy_env_instance.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=target_action_shape, dtype=np.float64)
                            temp_env_for_load = DummyVecEnv([lambda: dummy_env_instance])
                        
                        model = rl_class.load(model_path, env=temp_env_for_load, device="cpu") # Load on CPU

                        obs_dim = model.policy.observation_space.shape
                        action_dim = model.policy.action_space.shape
                        
                        print(f"  - {algo_name}:")
                        print(f"    Dimensione Input (Observation Space): {obs_dim}")
                        print(f"    Dimensione Output (Action Space): {action_dim}")
                        
                        # Attempt to get hidden layer info if available
                        if hasattr(model.policy, 'mlp_extractor'):
                            print(f"    Architettura della rete (strati nascosti): Presente (dettagli specifici dipendono dall'implementazione).")
                        
                        if temp_env_for_load:
                            temp_env_for_load.close()

                    except Exception as e:
                        print(f"  - ERRORE durante il caricamento o l'ispezione del modello {algo_name}: {e}")
                else:
                    print(f"  - Modello {algo_name} non trovato in {model_dir}")

        # --- Number of Simulations for Benchmark ---
    num_sims = int(get_interactive_input("Quante simulazioni di valutazione per scenario?", "1"))

    # --- Run Final Benchmark ---
    run_benchmark(
        config_files=benchmark_scenarios,
        reward_func=selected_reward_func,
        algorithms_to_run=algorithms_to_run, # <-- Here we use the full list with Optimal (if chosen)
        num_simulations=num_sims,
        model_dir=model_dir,
        is_multi_scenario=is_multi_scenario,
        price_data_file=selected_price_file_abs_path
    )

    print("\n--- ESECUZIONE COMPLETATA ---")

if __name__ == "__main__":
    main()
