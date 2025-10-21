import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import glob
import random
import yaml
from tqdm import tqdm
import multiprocessing
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.pulp_mpc import OnlineMPC_Solver

# =====================================================================================
# --- PARAMETRI DI ADDESTRAMENTO E RETE ---
# =====================================================================================
CONFIG_PATH = 'ev2gym/example_config_files/'
AVAILABLE_SCENARIOS = glob.glob(os.path.join(CONFIG_PATH, "*.yaml"))

if not AVAILABLE_SCENARIOS:
    raise FileNotFoundError(f"Nessun file di scenario .yaml trovato in '{CONFIG_PATH}'")

MAX_CS = 0
for scenario_file in AVAILABLE_SCENARIOS:
    with open(scenario_file, 'r') as f:
        config = yaml.safe_load(f)
        if 'number_of_charging_stations' in config:
            MAX_CS = max(MAX_CS, config['number_of_charging_stations'])

if MAX_CS == 0:
    raise ValueError("Impossibile determinare il numero massimo di stazioni di ricarica.")

print(f"Trovati {len(AVAILABLE_SCENARIOS)} scenari. Max CS: {MAX_CS}")

# Parametri
NUM_SAMPLES = 300
CONTROL_HORIZON = 5
MODEL_SAVE_PATH = 'ev2gym/baselines/mpc_approximator_nn.pth'
STATE_VECTOR_SIZE = MAX_CS * 2 + CONTROL_HORIZON * 2  # socs, departures, prices_ch, prices_dis

# Parametri della Rete Neurale
HIDDEN_LAYERS = [256, 128, 64]
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 64

# =====================================================================================
# --- DEFINIZIONE DELLA RETE NEURALE ---
# =====================================================================================
class MPCApproximatorNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(MPCApproximatorNet, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =====================================================================================
# --- FUNZIONI WORKER PER LA PARALLELIZZAZIONE ---
# =====================================================================================
def build_state_vector(env, max_cs, horizon):
    current_step = env.current_step
    num_cs_in_env = env.cs
    
    ev_socs = np.zeros(max_cs)
    ev_time_to_departure = np.zeros(max_cs)
    
    for i in range(num_cs_in_env):
        ev = next((ev for ev in env.charging_stations[i].evs_connected if ev is not None), None)
        if ev:
            ev_socs[i] = ev.get_soc()
            ev_time_to_departure[i] = max(0, ev.time_of_departure - current_step)

    h = min(horizon, env.simulation_length - current_step)
    prices_charge = env.charge_prices[0, current_step : current_step + h]
    prices_discharge = env.discharge_prices[0, current_step : current_step + h]
    
    padded_prices_ch = np.pad(prices_charge, (0, horizon - len(prices_charge)), 'edge')
    padded_prices_dis = np.pad(prices_discharge, (0, horizon - len(prices_discharge)), 'edge')

    return np.concatenate([ev_socs, ev_time_to_departure, padded_prices_ch, padded_prices_dis])

def generate_sample(_):
    """
    Genera un singolo campione (stato, azione) risolvendo il problema MPC.
    Questa funzione è progettata per essere eseguita in un processo separato.
    """
    env = None
    try:
        # Seleziona uno scenario casuale e crea l'ambiente
        selected_config = random.choice(AVAILABLE_SCENARIOS)
        env = EV2Gym(config_file=selected_config, generate_rnd_game=True)
        
        # Salta se la simulazione è troppo corta
        if env.simulation_length <= CONTROL_HORIZON + 1:
            return None

        # Scegli uno step casuale
        random_step = np.random.randint(0, env.simulation_length - CONTROL_HORIZON - 1)
        
        # Avanza l'ambiente fino allo step casuale
        env.reset()
        for _ in range(random_step):
            env.step(np.zeros(env.cs))

        # Se non ci sono EV connessi, il campione non è valido
        if not any(ev is not None for cs in env.charging_stations for ev in cs.evs_connected):
            return None

        # Risolvi l'MPC per ottenere l'azione ottimale (oracolo)
        mpc_oracle = OnlineMPC_Solver(env, control_horizon=CONTROL_HORIZON)
        state_vector = build_state_vector(env, MAX_CS, CONTROL_HORIZON)
        action_normalized = mpc_oracle.get_action(env)
        
        # Se l'azione è significativa, calcola le potenze e restituisci il campione
        if np.any(action_normalized):
            optimal_powers = np.zeros(MAX_CS)
            for i in range(env.cs):
                max_power = env.charging_stations[i].get_max_power()
                optimal_powers[i] = action_normalized[i] * max_power
            return state_vector, optimal_powers
            
    except Exception as e:
        # print(f"Errore in un processo worker: {e}")
        return None
    finally:
        if env:
            env.close()
    return None

# =====================================================================================
# --- SCRIPT PRINCIPALE DI ADDESTRAMENTO ---
# =====================================================================================
if __name__ == "__main__":
    print("\n--- Avvio generazione dataset PARALLELIZZATA per MPC Esplicito Approssimato (NN) ---")
    
    X_data, y_data = [], []
    
    # Usa multiprocessing.Pool per parallelizzare la generazione dei dati
    # Usa un numero di processi pari ai core della CPU meno uno, o 1 se c'è un solo core
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Utilizzando {num_processes} processi worker.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=NUM_SAMPLES, desc="Campionamento Stati Multi-Scenario") as pbar:
            # imap_unordered è efficiente perché processa i risultati appena sono pronti
            for result in pool.imap_unordered(generate_sample, range(NUM_SAMPLES)):
                if result is not None:
                    state, action = result
                    X_data.append(state)
                    y_data.append(action)
                    pbar.update(1)

    if not X_data:
        raise ValueError("Nessun dato di addestramento generato. Controlla la logica di generazione o aumenta NUM_SAMPLES.")

    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32)
    
    print(f"\nDataset generato. Shape X: {X_tensor.shape}, Shape y: {y_tensor.shape}")

    print("--- Avvio addestramento della Rete Neurale ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MPCApproximatorNet(STATE_VECTOR_SIZE, MAX_CS, HIDDEN_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_epoch_loss:.4f}")

    print("Addestramento completato.")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"--- Modello salvato con successo in '{MODEL_SAVE_PATH}' ---")
