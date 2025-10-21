import os
import inspect
from glob import glob
import subprocess
import time
import yaml
import json
from collections import defaultdict

# Importazioni da librerie di RL
from stable_baselines3 import SAC, DDPG, TD3, PPO
from sb3_contrib import TQC

# Importazioni dalla libreria custom ev2gym e da altri script
from run_experiments import (
    calculate_max_cs,
    get_algorithms,
    train_rl_models_if_requested,
    run_benchmark
)
from ev2gym.rl_agent import reward as reward_module

# --- Funzioni di utilità per l'interfaccia utente ---

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
        print("Nessun elemento disponibile per la selezione.")
        return [] if multiple else None

    if multiple:
        choices_str = get_interactive_input(f"Seleziona uno o più (es. '1 3', 'tutti')", 'tutti').lower()
        if 'tutti' in choices_str:
            return items
        try:
            indices = [int(i) - 1 for i in choices_str.split()]
            return [items[i] for i in indices if 0 <= i < len(items)]
        except (ValueError, IndexError):
            print("Selezione non valida. Verranno usati tutti gli elementi.")
            return items
    else:
        try:
            choice_str = get_interactive_input(f"Scelta", str(default_choice))
            choice = int(choice_str) - 1
            if 0 <= choice < len(items):
                return items[choice]
            else:
                raise IndexError
        except (ValueError, IndexError):
            print(f"Selezione non valida. Verrà usata la scelta di default ({default_choice}).")
            return items[default_choice - 1]

def get_available_rl_algorithms():
    """
    Restituisce un dizionario di algoritmi RL compatibili con spazi di azione continui.
    DQN e altri algoritmi per azioni discrete sono esclusi.
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

def select_model_directory(prompt="Seleziona il set di modelli da caricare:"):
    """Lets the user select a directory from ./saved_models/."""
    saved_models_dir = './saved_models/'
    if not os.path.exists(saved_models_dir) or not os.listdir(saved_models_dir):
        print(f"\nERRORE: Nessuna cartella trovata in '{saved_models_dir}'. Esegui prima l'addestramento.")
        return None, False

    available_models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
    selected_model_name = select_from_list(available_models, prompt, multiple=False)
    
    if not selected_model_name:
        return None, False

    model_dir = os.path.join(saved_models_dir, selected_model_name)
    
    # La modalità multi-scenario è ora lo standard
    is_multi_scenario = True
    print(f"\nModelli selezionati da: {model_dir} (modalità multi-scenario di default)")
    return model_dir, is_multi_scenario

# --- Flussi Principali: Addestramento e Plot ---

def run_training_flow():
    """Gestisce il flusso di lavoro per l'addestramento di nuovi modelli."""
    print("\n--- Inizio Flusso di Addestramento ---")

    # 1. Seleziona algoritmi da addestrare
    available_rl_algos = get_available_rl_algorithms()
    selected_algo_names = select_from_list(
        list(available_rl_algos.keys()), 
        "Seleziona gli algoritmi RL da addestrare:", 
        multiple=True
    )
    if not selected_algo_names:
        print("Nessun algoritmo selezionato. Addestramento annullato.")
        return

    algorithms_to_train = {k: available_rl_algos[k] for k in selected_algo_names}
    print(f"Algoritmi da addestrare: {list(algorithms_to_train.keys())}")

    # 2. Configurazione dell'addestramento (semplificata)
    config_path = "ev2gym/example_config_files/"
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    
    print("\nL'addestramento verrà eseguito in modalità 'Parametri Dinamici'.")
    print("Questa modalità addestra i modelli con parametri variati dinamicamente per una maggiore robustezza.")
    base_scenario_path = select_from_list(available_scenarios, "Seleziona uno scenario di BASE per la randomizzazione dei parametri:")
    
    steps_for_training = int(get_interactive_input("Per quanti passi di training totali?", "100000"))
    session_name = get_interactive_input("Inserisci un nome per questa sessione di addestramento", f"dynamic_{'_'.join(selected_algo_names).lower()}_{time.strftime('%Y%m%d')}")
    model_dir = f'./saved_models/{"" .join(c for c in session_name if c.isalnum() or c in ("_", "-")).rstrip()}/'
    os.makedirs(model_dir, exist_ok=True)

    # 3. Selezione reward e file prezzi
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    selected_reward_tuple = select_from_list(available_rewards, "Scegli la funzione di reward:", default_choice=1)
    selected_reward_func = selected_reward_tuple[1]

    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    default_price_index = available_price_files.index(default_price_file) + 1 if default_price_file in available_price_files else 1
    selected_price_file_name = select_from_list(available_price_files, "Seleziona il file CSV per i prezzi dell'energia:", default_choice=default_price_index)
    selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)

    # 4. Esecuzione dell'addestramento
    print(f"\n--- Inizio addestramento nella cartella: {model_dir} ---")
    train_rl_models_if_requested(
        scenarios_to_test=[base_scenario_path],
        selected_reward_func=selected_reward_func,
        algorithms_to_run=algorithms_to_train,
        is_multi_scenario=True, # Sempre True con la nuova modalità
        model_dir=model_dir,
        selected_price_file_abs_path=selected_price_file_abs_path,
        steps_for_training=steps_for_training,
        training_mode='dynamic', # Modalità di default robusta
        session_name=session_name
    )
    print("\n--- Addestramento Completato ---")

def run_plotting_flow():
    """Gestisce il flusso di lavoro per il benchmark e il plotting di modelli esistenti."""
    print("\n--- Inizio Flusso di Plotting ---")

    # 1. Seleziona la cartella dei modelli
    model_dir, is_multi_scenario = select_model_directory()
    if not model_dir:
        return

    # 2. Rileva algoritmi disponibili in quella cartella
    available_model_files = glob(os.path.join(model_dir, '*_model.zip'))
    trained_rl_algos = [os.path.basename(f).replace('_model.zip', '').replace('_', '+').upper() for f in available_model_files]
    
    if not trained_rl_algos:
        print(f"Nessun file modello .zip trovato in {model_dir}. Impossibile procedere.")
        return
        
    print(f"Algoritmi RL addestrati trovati in questa sessione: {trained_rl_algos}")

    # 3. Ottieni algoritmi di base (Euristiche, MPC) e RL
    MAX_CS = calculate_max_cs("ev2gym/example_config_files/")
    all_base_algos = get_algorithms(MAX_CS, is_thesis_mode=True)
    baselines = {k: v for k, v in all_base_algos.items() if v[1] is None}
    
    # Usa la nuova funzione dinamica per ottenere le definizioni degli algoritmi RL
    all_rl_definitions = get_available_rl_algorithms()
    available_rl_from_files = {k: v for k, v in all_rl_definitions.items() if k in trained_rl_algos}

    # Unisci le definizioni di tutti gli algoritmi disponibili per questo plot
    all_available_definitions = {**baselines, **available_rl_from_files}

    # 4. Seleziona algoritmi da plottare
    # L'ordine qui determina come appaiono nella lista di selezione
    plot_candidates = sorted(list(baselines.keys())) + sorted(list(available_rl_from_files.keys()))
    selected_for_plot = select_from_list(plot_candidates, "Seleziona gli algoritmi da confrontare nel benchmark:", multiple=True)
    
    if not selected_for_plot:
        print("Nessun algoritmo selezionato per il plot. Annullato.")
        return

    algorithms_to_run = {k: all_available_definitions[k] for k in selected_for_plot}
    print(f"\nAlgoritmi che verranno eseguiti nel benchmark: {list(algorithms_to_run.keys())}")

    # 5. Configurazione del benchmark
    config_path = "ev2gym/example_config_files/"
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    benchmark_scenarios = select_from_list(available_scenarios, "Seleziona gli scenari per il BENCHMARK:", multiple=True)
    
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    selected_reward_tuple = select_from_list(available_rewards, "Scegli la funzione di reward (assicurati sia la stessa dell'addestramento):", default_choice=1)
    selected_reward_func = selected_reward_tuple[1]

    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    default_price_index = available_price_files.index(default_price_file) + 1 if default_price_file in available_price_files else 1
    selected_price_file_name = select_from_list(available_price_files, "Seleziona il file CSV per i prezzi dell'energia:", default_choice=default_price_index)
    selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)

    num_sims = int(get_interactive_input("Quante simulazioni di valutazione per scenario?", "1"))

    # 6. Esecuzione del benchmark
    print("\n--- Inizio Benchmark e Generazione Grafici ---")
    run_benchmark(
        config_files=benchmark_scenarios,
        reward_func=selected_reward_func,
        algorithms_to_run=algorithms_to_run,
        num_simulations=num_sims,
        model_dir=model_dir,
        is_multi_scenario=is_multi_scenario,
        price_data_file=selected_price_file_abs_path
    )
    print("\n--- ESECUZIONE COMPLETATA ---")


def main():
    """Funzione principale che orchestra l'esecuzione."""
    
    # --- Esecuzione preliminare di Fit_battery.py ---
    if get_interactive_input("Vuoi eseguire 'Fit_battery.py' per calibrare il modello di degradazione? (s/n)", "n").lower() == 's':
        print("--- Esecuzione di Fit_battery.py ---")
        try:
            subprocess.run(["python", "Fit_battery.py"], check=True)
            print("--- Fit_battery.py completato. ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERRORE: {e}. Lo script procederà con i parametri esistenti.")

    # --- Menu Principale ---
    while True:
        choice = get_interactive_input("\nCosa vuoi fare?\n  1. Addestrare nuovi modelli RL\n  2. Eseguire benchmark e plottare risultati di modelli esistenti\n\nScelta", "2")
        if choice == '1':
            run_training_flow()
        elif choice == '2':
            run_plotting_flow()
        else:
            print("Scelta non valida.")

        if get_interactive_input("\nVuoi eseguire un'altra operazione? (s/n)", "n").lower() != 's':
            break
            
    print("\n--- Programma terminato. ---")

if __name__ == "__main__":
    main()