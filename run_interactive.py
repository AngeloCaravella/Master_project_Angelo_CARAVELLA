import os
import inspect
from glob import glob
import subprocess

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

def select_from_list(items, prompt, multiple=False):
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
        choice = input(f"Scelta (default: 1): ") or '1'
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

    # --- Select Scenarios ---
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    scenarios_to_test = select_from_list(available_scenarios, "Seleziona gli scenari da testare:", multiple=True)
    print(f"Scenari selezionati: {[os.path.basename(s) for s in scenarios_to_test]}")

    # --- Select Reward Function ---
    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    reward_names = [name for name, _ in available_rewards]
    selected_reward_name = select_from_list(reward_names, "Scegli la funzione di reward:")
    selected_reward_func = next(func for name, func in available_rewards if name == selected_reward_name)
    print(f"Funzione di reward selezionata: {selected_reward_name}")

    # --- Select Price File ---
    price_data_dir = os.path.join(os.path.dirname(__file__), 'ev2gym', 'data')
    available_price_files = sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])
    default_price_file = "Netherlands_day-ahead-2015-2024.csv"
    selected_price_file_name = select_from_list(available_price_files, "Seleziona il file CSV per i prezzi dell'energia:")
    if selected_price_file_name == default_price_file:
        selected_price_file_abs_path = os.path.join(price_data_dir, default_price_file)
    else:
        selected_price_file_abs_path = os.path.join(price_data_dir, selected_price_file_name)
    print(f"File prezzi selezionato: {selected_price_file_name}")

    # --- Train RL Models ---
    train_rl_models = get_interactive_input("Vuoi addestrare i modelli RL? (s/n)", "n").lower() == 's'
    steps_for_training = 0
    if train_rl_models:
        steps_for_training = int(get_interactive_input("Per quanti passi di training?", "100000"))

    # --- Number of Simulations ---
    num_sims = int(get_interactive_input("Quante simulazioni di valutazione per scenario?", "1"))

    # --- Run Simulation ---
    is_multi_scenario = len(scenarios_to_test) > 1
    scenario_name_for_path = 'multi_scenario' if is_multi_scenario else os.path.basename(scenarios_to_test[0]).replace(".yaml", "")
    model_dir = f'./saved_models/{scenario_name_for_path}/'
    os.makedirs(model_dir, exist_ok=True)

    if train_rl_models:
        train_rl_models_if_requested(
            scenarios_to_test=scenarios_to_test,
            selected_reward_func=selected_reward_func,
            algorithms_to_run=algorithms_to_run,
            is_multi_scenario=is_multi_scenario,
            model_dir=model_dir,
            selected_price_file_abs_path=selected_price_file_abs_path,
            steps_for_training=steps_for_training
        )

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
