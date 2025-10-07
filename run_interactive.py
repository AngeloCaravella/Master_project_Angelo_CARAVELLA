
import os
import inspect
from glob import glob
import subprocess
import time

# Import functions from run_experiments
from run_experiments import (
    calculate_max_cs,
    get_algorithms,
    train_rl_models_if_requested,
    run_benchmark
)
from ev2gym.rl_agent import reward as reward_module

def get_interactive_input(prompt, default=None):
    """Helper function to get user input with a default value."""
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input or default

def select_from_list(items, prompt, multiple=False, default_choice=1):
    """Helper function to let the user select one or more items from a list."""
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"{i+1}. {os.path.basename(item).replace('.yaml', '')}")
    
    if multiple:
        choices = input(f"Seleziona uno o più scenari (es. '1 3', 'tutti') (default: tutti): ").lower() or 'tutti'
        if 'tutti' in choices:
            return items
        else:
            return [items[int(i)-1] for i in choices.split()]
    else:
        choice = input(f"Scelta (default: {default_choice}): ") or str(default_choice)
        return items[int(choice)-1]

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
    plot_mode_choice = get_interactive_input("Scegli modalità grafici: '1' per Tesi, '2' per Completa", "1")
    is_thesis_mode = (plot_mode_choice == '1')

    # --- Calculate MAX_CS ---
    config_path = "ev2gym/example_config_files/"
    MAX_CS = calculate_max_cs(config_path)
    print(f"\nRilevato un massimo di {MAX_CS} stazioni di ricarica tra tutti gli scenari.")

    # --- MPC Type ---
    mpc_type_choice = get_interactive_input("Scegli il tipo di MPC: 'linear' o 'quadratic'", "linear")

    # --- Get Algorithms ---
    algorithms_to_run = get_algorithms(MAX_CS, is_thesis_mode, mpc_type_choice)
    print(f"\nAlgoritmi che verranno eseguiti: {list(algorithms_to_run.keys())}")

    # --- Configurazione specifica per DDPG+PER ---
    if 'DDPG+PER' in algorithms_to_run:
        print("\n--- Configurazione Prioritized Experience Replay (PER) per DDPG+PER ---")
        print("Alpha controlla il livello di priorità (0=uniforme, 1=massima priorità).")
        print("Beta corregge il bias introdotto dalla priorità (aumenta fino a 1).")
        per_alpha = float(get_interactive_input("Inserisci il valore di alpha", "0.6"))
        per_beta = float(get_interactive_input("Inserisci il valore iniziale di beta", "0.4"))
        
        # Aggiungi i parametri specifici per il replay buffer
        ddpg_per_kwargs = algorithms_to_run['DDPG+PER'][2]
        ddpg_per_kwargs['replay_buffer_kwargs'] = {
            'alpha': per_alpha,
            'beta': per_beta
        }

    # --- Configurazione specifica per OnlineMPC_Solver ---
    online_mpc_keys = [k for k in algorithms_to_run if 'Online_MPC' in k]
    if online_mpc_keys:
        print("\n--- Configurazione MPC Online (per orizzonte di controllo > 1) ---")
        pred_h = int(get_interactive_input("Inserisci l'orizzonte di predizione (Np)", "5"))
        ctrl_h_input = get_interactive_input("Inserisci l'orizzonte di controllo (Nc) (es. 1, 3, o 'half')", "half")

        try:
            ctrl_h = int(ctrl_h_input)
        except ValueError:
            ctrl_h = ctrl_h_input

        for key in online_mpc_keys:
            mpc_kwargs = algorithms_to_run[key][2]
            mpc_kwargs['prediction_horizon'] = pred_h
            mpc_kwargs['control_horizon'] = ctrl_h
            print(f"MPC Online '{key}' aggiornato: Np={pred_h}, Nc={ctrl_h}")

    # --- Select Scenarios ---
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    scenarios_to_test = select_from_list(available_scenarios, "Seleziona gli scenari da testare:", multiple=True)
    print(f"Scenari selezionati: {[os.path.basename(s) for s in scenarios_to_test]}")

    # --- Select Reward Function ---
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    reward_names = [name for name, _ in available_rewards]
    selected_reward_name = select_from_list(reward_names, "Scegli la funzione di reward:", default_choice=3)
    selected_reward_func = next(func for name, func in available_rewards if name == selected_reward_name)
    print(f"Funzione di reward selezionata: {selected_reward_name}")

    # --- Select Price File ---
    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    # Find the index of the default file to pass to the function
    try:
        default_price_index = available_price_files.index(default_price_file) + 1
    except ValueError:
        default_price_index = 1 # Fallback to 1 if not found

    selected_price_file_name = select_from_list(available_price_files, "Seleziona il file CSV per i prezzi dell'energia:", default_choice=default_price_index)
    if selected_price_file_name == default_price_file:
        selected_price_file_abs_path = os.path.join(price_data_dir, default_price_file)
    else:
        selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)
    print(f"File prezzi selezionato: {selected_price_file_name}")

    # --- Train RL Models ---
    train_rl_models = get_interactive_input("Vuoi addestrare i modelli RL? (s/n)", "n").lower() == 's'
    steps_for_training = 0
    training_mode = 'single' # Default
    curriculum_steps_per_level = 10000 # Default

    # =================================================================================
    # --- INIZIO DELLA CORREZIONE ---
    # Inizializziamo 'is_multi_scenario' qui, basandoci sul numero di scenari
    # selezionati per il benchmark. Questo assicura che la variabile esista sempre,
    # anche se il blocco di addestramento RL viene saltato.
    is_multi_scenario = len(scenarios_to_test) > 1
    # --- FINE DELLA CORREZIONE ---
    # =================================================================================

    if train_rl_models:
        prompt = ("Scegli la modalità di addestramento:\n" 
                  "  '1' per Scenario Singolo\n"
                  "  '2' per Multi-Scenario Casuale (con reinserimento)\n"
                  "  '3' per Curriculum Learning\n"
                  "  '4' per Casuale a Epoche (senza reinserimento)")
        mode_choice = get_interactive_input(prompt, "2")
        
        if mode_choice == '1':
            training_mode = 'single'
            scenarios_to_test = [select_from_list(available_scenarios, "Seleziona lo scenario per l'addestramento:")]
            is_multi_scenario = False
        elif mode_choice == '2':
            training_mode = 'random'
            scenarios_to_test = select_from_list(available_scenarios, "Seleziona gli scenari per l'addestramento multi-scenario:", multiple=True)
            is_multi_scenario = True
        elif mode_choice == '3':
            training_mode = 'curriculum'
            print("\n--- Configurazione Curriculum Learning ---")
            print("Scenari disponibili:")
            for i, s in enumerate(available_scenarios):
                print(f"{i+1}. {os.path.basename(s).replace('.yaml', '')}")
            
            order_input = input("Specifica l'ordine degli scenari (es. '1 3 2' per usare il primo, poi il terzo, poi il secondo): ")
            order = [int(i)-1 for i in order_input.split()]
            scenarios_to_test = [available_scenarios[i] for i in order]
            
            print("\nCurriculum definito con il seguente ordine:")
            for i, s in enumerate(scenarios_to_test):
                print(f"{i+1}. {os.path.basename(s)}")
            
            curriculum_steps_per_level = int(get_interactive_input("Quanti passi di training per ogni livello del curriculum?", "10000"))
            is_multi_scenario = True
        elif mode_choice == '4':
            training_mode = 'shuffled'
            scenarios_to_test = select_from_list(available_scenarios, "Seleziona gli scenari per l'addestramento Casuale a Epoche:", multiple=True)
            is_multi_scenario = True

        steps_for_training = int(get_interactive_input("Per quanti passi di training totali?", "100000"))

        # Ask for a session name for multi-scenario/curriculum trainings
        if is_multi_scenario:
            default_name = f"{training_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
            session_name = get_interactive_input(
                "Inserisci un nome per questa sessione di addestramento (verrà salvata in 'saved_models/')",
                default_name
            )
            # Sanitize the name to be a valid folder name
            scenario_name_for_path = "".join(c for c in session_name if c.isalnum() or c in ('_', '-')).rstrip()
        else:
            scenario_name_for_path = os.path.basename(scenarios_to_test[0]).replace(".yaml", "")
        
        model_dir = f'./saved_models/{scenario_name_for_path}/'
        os.makedirs(model_dir, exist_ok=True)

        train_rl_models_if_requested(
            scenarios_to_test=scenarios_to_test,
            selected_reward_func=selected_reward_func,
            algorithms_to_run=algorithms_to_run,
            is_multi_scenario=is_multi_scenario,
            model_dir=model_dir,
            selected_price_file_abs_path=selected_price_file_abs_path,
            steps_for_training=steps_for_training,
            training_mode=training_mode,
            curriculum_steps_per_level=curriculum_steps_per_level
        )
    else:
        # --- Select Pre-trained Model ---
        saved_models_dir = './saved_models/'
        if not os.path.exists(saved_models_dir):
            os.makedirs(saved_models_dir)
        
        available_models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
        
        if not available_models:
            print("\nERRORE: Nessun modello addestrato trovato in './saved_models/'.")
            print("Impossibile eseguire il benchmark senza modelli. Esegui prima l'addestramento.")
            return

        selected_model_name = select_from_list(available_models, "Seleziona il set di modelli da caricare:", multiple=False)
        
        model_dir = os.path.join(saved_models_dir, selected_model_name)
        
        # Infer is_multi_scenario from the folder name
        if 'multi' in selected_model_name.lower() or 'curriculum' in selected_model_name.lower():
            is_multi_scenario = True
        else:
            is_multi_scenario = False
        
        print(f"\nModelli selezionati da: {model_dir}")
        print(f"Modalità rilevata: {'multi-scenario' if is_multi_scenario else 'scenario singolo'}")
        print("ATTENZIONE: Assicurati che gli scenari di test selezionati siano compatibili con i modelli caricati.")

    # --- Number of Simulations ---
    num_sims = int(get_interactive_input("Quante simulazioni di valutazione per scenario?", "1"))

    # --- Run Simulation ---
    run_benchmark(
        config_files=scenarios_to_test,
        reward_func=selected_reward_func,
        algorithms_to_run=algorithms_to_run,
        num_simulations=num_sims,
        model_dir=model_dir,
        is_multi_scenario=is_multi_scenario,
        price_data_file=selected_price_file_abs_path
    )

    print("\n--- ESECUZIONE COMPLETATA ---")

if __name__ == "__main__":
    main()
