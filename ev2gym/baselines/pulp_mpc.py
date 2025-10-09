# --- START OF FILE pulp_mpc.py ---

import pulp
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from pathlib import Path

# =====================================================================================
# --- SOLVER 1: MPC CON PRIORITÀ ALLA SODDISFAZIONE UTENTE (Vincoli Morbidi) ---
# =====================================================================================/
#

# =====================================================================================
# --- SOLVER 2: MPC PER V2G PROFIT MAXIMIZATION (come da Paper, Vincoli Rigidi) ---
# =====================================================================================
#V2GProfitMaxMPC_Solver
class OnlineMPC_Solver:
    """
    Un solver MPC con un unico obiettivo: garantire la ricarica dell'utente.
    
    STRATEGIA:
    1. OBIETTIVO: Minimizzare il costo di acquisto dell'energia.
    2. DIVIETO DI SCARICA (V2G): La scarica è completamente disabilitata. Il solver
       può solo decidere QUANDO caricare e QUANTO caricare.
    3. VINCOLO RIGIDO SULLA CARICA: È obbligatorio raggiungere il target di SoC
       alla partenza del veicolo.

    Questo approccio elimina ogni conflitto con la massimizzazione del profitto e
    forza il solver a comportarsi come un puro fornitore di servizi di ricarica intelligente.
    """
    def __init__(self, env, prediction_horizon=25, control_horizon='half', 
                 mpc_desired_soc_factor=0.90,
                 penalty_overload=100.0,
                 **kwargs):
        
        self.env = env
        self.H = prediction_horizon
        self.Nc = max(1, self.H // 2) if control_horizon == 'half' else int(control_horizon)
        self.mpc_desired_soc_factor = mpc_desired_soc_factor
        self.penalty_overload = penalty_overload
        
        print(f"MPC (Charge-Only, User-First) configurato con Np={self.H}, Nc={self.Nc}")
        print(f"    -> Target SoC alla partenza: {self.mpc_desired_soc_factor:.0%}")

        self.action_plan = []
        self.plan_step = 0

    def get_action(self, env):
        if self.plan_step < len(self.action_plan):
            action_to_take = self.action_plan[self.plan_step]
            self.plan_step += 1
            return action_to_take

        self.action_plan, self.plan_step = [], 0
        current_step, sim_length, num_cs = env.current_step, env.simulation_length, env.cs
        timescale_h = env.timescale / 60.0
        prediction_horizon = min(self.H, sim_length - current_step)
        if prediction_horizon <= 0: return np.zeros(num_cs)

        transformer = env.transformers[0]
        transformer_limit_horizon = transformer.get_power_limits(current_step, prediction_horizon)
        load_forecast, pv_forecast = transformer.get_load_pv_forecast(current_step, prediction_horizon)

        E_initial, active_evs = np.zeros(num_cs), {}
        for i in range(num_cs):
            cs = env.charging_stations[i]
            ev = next((ev for ev in cs.evs_connected if ev is not None), None)
            if ev:
                eta_ch = np.mean(list(ev.charge_efficiency.values())) if isinstance(ev.charge_efficiency, dict) else ev.charge_efficiency
                active_evs[i] = {'ev': ev, 'eta_ch': eta_ch}
                E_initial[i] = ev.get_soc() * ev.battery_capacity

        # --- OBIETTIVO: MINIMIZZARE IL COSTO ---
        prob = pulp.LpProblem(f"ChargeOnly_CostMin_MPC_{current_step}", pulp.LpMinimize)
        
        indices = [(i, j) for i in range(num_cs) for j in range(prediction_horizon)]
        P_ch = pulp.LpVariable.dicts("ChargePower", indices, lowBound=0)
        E = pulp.LpVariable.dicts("Energy", indices, lowBound=0)
        slack_overload = pulp.LpVariable.dicts("SlackOverload", range(prediction_horizon), lowBound=0)

        prices_charge = env.charge_prices[0, current_step : current_step + prediction_horizon]
        
        # L'obiettivo è minimizzare il costo dell'energia acquistata + la penalità di sovraccarico
        cost_of_charging = pulp.lpSum(
            prices_charge[t] * P_ch[i, t] * timescale_h
            for i in active_evs.keys() for t in range(prediction_horizon)
        )
        overload_penalty = pulp.lpSum(self.penalty_overload * slack_overload[t] for t in range(prediction_horizon))
        prob.setObjective(cost_of_charging + overload_penalty)

        # --- VINCOLI DEL MODELLO ---
        for cs_id, data in active_evs.items():
            ev, eta_ch = data['ev'], data['eta_ch']
            
            for t in range(prediction_horizon):
                # Vincolo sulla potenza massima (solo carica)
                prob += P_ch[cs_id, t] <= ev.max_ac_charge_power
                
                # Dinamica della batteria (solo carica)
                E_prev = E_initial[cs_id] if t == 0 else E[cs_id, t-1]
                prob += E[cs_id, t] == E_prev + (P_ch[cs_id, t] * eta_ch) * timescale_h
                
                # Limiti fisici della batteria
                prob += E[cs_id, t] >= ev.min_battery_capacity
                prob += E[cs_id, t] <= ev.battery_capacity
            
            # --- VINCOLO RIGIDO (HARD CONSTRAINT) PER LA SODDISFAZIONE UTENTE ---
            departure_step_in_horizon = ev.time_of_departure - current_step - 1
            if 0 <= departure_step_in_horizon < prediction_horizon:
                desired_energy = ev.desired_capacity * self.mpc_desired_soc_factor
                prob += E[cs_id, departure_step_in_horizon] >= desired_energy

        for i in range(num_cs):
            if i not in active_evs:
                for t in range(prediction_horizon):
                    prob += P_ch[i, t] == 0

        # Vincolo morbido per il limite del trasformatore (solo carica)
        for t in range(prediction_horizon):
            power_evs = pulp.lpSum(P_ch[i, t] for i in range(num_cs))
            total_power = power_evs + load_forecast[t] + pv_forecast[t]
            limit = transformer_limit_horizon[t] if t < len(transformer_limit_horizon) else transformer_limit_horizon[-1]
            prob += total_power <= limit + slack_overload[t]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            new_plan = []
            effective_Nc = min(self.Nc, prediction_horizon)
            for t in range(effective_Nc):
                action_t = np.zeros(num_cs)
                for i in range(num_cs):
                    charge = pulp.value(P_ch[i, t])
                    net_power = charge or 0
                    max_power = env.charging_stations[i].get_max_power()
                    if max_power > 0: action_t[i] = net_power / max_power
                new_plan.append(np.clip(action_t, 0, 1)) # L'azione ora è solo positiva
            self.action_plan = new_plan
            if self.action_plan:
                self.plan_step = 1
                return self.action_plan[0]
        else:
            print(f"Attenzione: MPC INFEASIBLE al timestep {current_step}. Impossibile raggiungere il target di ricarica.")
            # Come fallback, carichiamo il più velocemente possibile
            action_fallback = np.ones(num_cs)
            return action_fallback
# =====================================================================================
# --- CLASSI PER MPC ESPLICITO APPROSSIMATO (Invariate) ---
# =====================================================================================

class ApproximateExplicitMPC:
    """
    Implementa un controller MPC Esplicito Approssimato.
    Questo approccio utilizza un modello di machine learning (es. Gradient Boosting)
    per approssimare la funzione di controllo ottimale calcolata dall'MPC online.
    """
    def __init__(self, env, model_path=None, control_horizon=10, max_cs=None, **kwargs):
        print(f"Inizializzazione controller MPC Esplicito Approssimato...")
        
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'mpc_approximator.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato in '{model_path}'. Esegui prima lo script di addestramento.")
        
        if max_cs is None:
            raise ValueError("Il parametro 'max_cs' (numero massimo di stazioni) deve essere fornito.")
        self.max_cs = max_cs

        self.model = joblib.load(model_path)
        self.H = control_horizon
        print(f"Modello caricato con successo da: {model_path}")

    def _build_state_vector(self, env):
        """
        Costruisce il vettore di stato per il modello approssimato.
        """
        current_step = env.current_step
        horizon = min(self.H, env.simulation_length - current_step)
        num_cs_in_env = env.cs
        
        ev_socs = np.zeros(self.max_cs)
        ev_time_to_departure = np.zeros(self.max_cs)
        
        for i in range(num_cs_in_env):
            ev = next((ev for ev in env.charging_stations[i].evs_connected if ev is not None), None)
            if ev:
                ev_socs[i] = ev.get_soc()
                ev_time_to_departure[i] = max(0, ev.time_of_departure - current_step)

        prices_charge = env.charge_prices[0, current_step : current_step + horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + horizon]
        
        padded_prices_ch = np.pad(prices_charge, (0, self.H - len(prices_charge)), 'edge')
        padded_prices_dis = np.pad(prices_discharge, (0, self.H - len(prices_discharge)), 'edge')

        state_vector = np.concatenate([
            ev_socs,
            ev_time_to_departure,
            padded_prices_ch,
            padded_prices_dis
        ])
        return state_vector.reshape(1, -1)

    def get_action(self, env):
        if env.current_step >= env.simulation_length - 1:
            return np.zeros(env.cs)
            
        state_vector = self._build_state_vector(env)
        predicted_powers = self.model.predict(state_vector)[0]
        
        action = np.zeros(env.cs)
        for i in range(env.cs):
            max_power = env.charging_stations[i].get_max_power()
            if max_power > 0:
                action[i] = predicted_powers[i] / max_power

        return np.clip(action, -1, 1)

class MPCApproximatorNet(nn.Module):
    """Definizione dell'architettura della rete neurale per l'approssimatore MPC."""
    def __init__(self, input_size, output_size, hidden_layers=[256, 128, 64]):
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

class ApproximateExplicitMPC_NN:
    """
    Implementa un controller MPC Esplicito Approssimato usando una Rete Neurale.
    """
    def __init__(self, env, model_path=None, control_horizon=5, max_cs=None, **kwargs):
        print("Inizializzazione controller MPC Esplicito Approssimato (Rete Neurale)...")
        
        if model_path is None:
            script_dir = Path(__file__).parent.resolve()
            model_path = script_dir / 'mpc_approximator_nn.pth'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato in '{model_path}'. Esegui prima lo script di addestramento.")
        
        if max_cs is None:
            raise ValueError("Il parametro 'max_cs' deve essere fornito.")
        self.max_cs = max_cs
        self.H = control_horizon

        state_vector_size = self.max_cs * 2 + self.H * 2
        output_size = self.max_cs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MPCApproximatorNet(input_size=state_vector_size, output_size=output_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"Modello Rete Neurale caricato con successo da: {model_path} su device {self.device}")

    def _build_state_vector(self, env):
        current_step = env.current_step
        horizon = min(self.H, env.simulation_length - current_step)
        num_cs_in_env = env.cs
        
        ev_socs = np.zeros(self.max_cs)
        ev_time_to_departure = np.zeros(self.max_cs)
        
        for i in range(num_cs_in_env):
            ev = next((ev for ev in env.charging_stations[i].evs_connected if ev is not None), None)
            if ev:
                ev_socs[i] = ev.get_soc()
                ev_time_to_departure[i] = max(0, ev.time_of_departure - current_step)

        prices_charge = env.charge_prices[0, current_step : current_step + horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + horizon]
        
        padded_prices_ch = np.pad(prices_charge, (0, self.H - len(prices_charge)), 'edge')
        padded_prices_dis = np.pad(prices_discharge, (0, self.H - len(prices_discharge)), 'edge')

        state_vector = np.concatenate([
            ev_socs,
            ev_time_to_departure,
            padded_prices_ch,
            padded_prices_dis
        ])
        return state_vector

    def get_action(self, env):
        if env.current_step >= env.simulation_length - 1:
            return np.zeros(env.cs)
            
        state_vector_np = self._build_state_vector(env)
        state_tensor = torch.tensor(state_vector_np, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predicted_powers_tensor = self.model(state_tensor)
        
        predicted_powers = predicted_powers_tensor.cpu().numpy()
        
        action = np.zeros(env.cs)
        for i in range(env.cs):
            max_power = env.charging_stations[i].get_max_power()
            if max_power > 0:
                action[i] = predicted_powers[i] / max_power

        return np.clip(action, -1, 1)

    
class OptimalOfflineSolver:
    """
    Un solver di ottimizzazione globale offline che calcola il piano di azione ottimale
    per l'intera simulazione, assumendo una conoscenza perfetta di tutti gli eventi futuri.
    
    Questo solver serve come benchmark teorico (upper bound) per la massimizzazione del profitto.
    """
    def __init__(self, env, mpc_desired_soc_factor=0.95, penalty_overload=1000.0, **kwargs):
        self.env = env
        self.mpc_desired_soc_factor = mpc_desired_soc_factor
        self.penalty_overload = penalty_overload
        
        # Il piano di azione verrà calcolato una sola volta
        self.action_plan = None
        print(f"Solver 'Optimal Offline' inizializzato.")

    def _solve_once(self, env):
        """
        Metodo principale che formula e risolve il problema di ottimizzazione globale.
        """
        print("--- [Optimal Solver] Avvio calcolo del piano di azione globale... ---")
        
        # 1. Raccogliere tutte le informazioni a priori dall'ambiente
        T = env.simulation_length
        num_cs = env.cs
        timescale_h = env.timescale / 60.0
        
        # Prezzi, carichi e generazione per l'intera simulazione
        prices_charge = env.charge_prices[0, :T]
        prices_discharge = env.discharge_prices[0, :T]
        transformer = env.transformers[0]
        load_forecast, pv_forecast = transformer.get_load_pv_forecast(0, T)
        transformer_limit = transformer.get_power_limits(0, T)

        # Mappa di tutti i veicoli che arriveranno, indicizzati per ID univoco
        # L'ambiente caricato da replay contiene questa informazione
        all_ev_sessions = env.get_all_future_ev_sessions()

        # 2. Impostare il problema di ottimizzazione con PuLP
        prob = pulp.LpProblem("Optimal_V2G_Profit_Maximization", pulp.LpMaximize)

        # 3. Definire le variabili di decisione per l'intera simulazione
        indices = [(i, t) for i in range(num_cs) for t in range(T)]
        P_ch = pulp.LpVariable.dicts("ChargePower", indices, lowBound=0)
        P_dis = pulp.LpVariable.dicts("DischargePower", indices, lowBound=0)
        
        # Variabili binarie per evitare carica e scarica simultanee
        b_ch = pulp.LpVariable.dicts("isCharging", indices, cat='Binary')
        b_dis = pulp.LpVariable.dicts("isDischarging", indices, cat='Binary')

        # Variabili di stato per l'energia di ogni EV
        ev_indices = [(ev_id, t) for ev_id in all_ev_sessions.keys() for t in range(T)]
        E = pulp.LpVariable.dicts("Energy", ev_indices, lowBound=0)
        
        # Variabile slack per la penalità di sovraccarico
        slack_overload = pulp.LpVariable.dicts("SlackOverload", range(T), lowBound=0)

        # 4. Definire la funzione obiettivo: Massimizzare il profitto
        profit = pulp.lpSum(
            (prices_discharge[t] * P_dis[i, t] - prices_charge[t] * P_ch[i, t]) * timescale_h
            for i, t in indices
        )
        penalty = pulp.lpSum(self.penalty_overload * slack_overload[t] for t in range(T))
        prob.setObjective(profit - penalty)

        # 5. Aggiungere i vincoli
        
        # Vincoli per ogni sessione EV
        for ev_id, ev_info in all_ev_sessions.items():
            ev = ev_info['ev']
            cs_id = ev_info['cs_id']
            t_arrival = ev_info['arrival_step']
            t_departure = ev_info['departure_step']
            
            eta_ch = np.mean(list(ev.charge_efficiency.values()))
            eta_dis = np.mean(list(ev.discharge_efficiency.values()))

            for t in range(t_arrival, t_departure):
                # Dinamica della batteria
                E_prev = ev.initial_capacity if t == t_arrival else E[ev_id, t-1]
                prob += E[ev_id, t] == E_prev + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h
                
                # Limiti di energia della batteria
                prob += E[ev_id, t] <= ev.battery_capacity
                prob += E[ev_id, t] >= ev.min_battery_capacity

            # Vincolo rigido sul SoC alla partenza
            desired_energy = ev.desired_capacity * self.mpc_desired_soc_factor
            prob += E[ev_id, t_departure - 1] >= desired_energy

        # Vincoli per ogni stazione di ricarica e timestep
        M = 1000 # Big-M per i vincoli binari
        for i, t in indices:
            # Mappa l'EV corretto alla stazione i al tempo t
            current_ev_id = None
            for ev_id, ev_info in all_ev_sessions.items():
                if ev_info['cs_id'] == i and ev_info['arrival_step'] <= t < ev_info['departure_step']:
                    current_ev_id = ev_id
                    break
            
            if current_ev_id:
                ev = all_ev_sessions[current_ev_id]['ev']
                # Limiti di potenza e vincoli binari
                prob += P_ch[i, t] <= ev.max_ac_charge_power * b_ch[i, t]
                prob += P_dis[i, t] <= ev.max_ac_discharge_power * b_dis[i, t]
                prob += b_ch[i, t] + b_dis[i, t] <= 1
            else:
                # Se non c'è nessun EV, la potenza è zero
                prob += P_ch[i, t] == 0
                prob += P_dis[i, t] == 0

        # Vincolo sul limite del trasformatore
        for t in range(T):
            power_evs = pulp.lpSum(P_ch[i, t] - P_dis[i, t] for i in range(num_cs))
            total_power = power_evs + load_forecast[t] - pv_forecast[t]
            prob += total_power <= transformer_limit[t] + slack_overload[t]

        # 6. Risolvere il problema
        prob.solve(pulp.PULP_CBC_CMD(msg=1))

        # 7. Estrarre la soluzione e creare l'action_plan
        if pulp.LpStatus[prob.status] == 'Optimal':
            print("--- [Optimal Solver] Soluzione ottimale globale trovata! ---")
            self.action_plan = np.zeros((T, num_cs))
            for t in range(T):
                for i in range(num_cs):
                    charge = pulp.value(P_ch[i, t])
                    discharge = pulp.value(P_dis[i, t])
                    net_power = charge - discharge
                    
                    max_power = env.charging_stations[i].get_max_power()
                    if max_power > 0:
                        self.action_plan[t, i] = net_power / max_power
            
            self.action_plan = np.clip(self.action_plan, -1, 1)
        else:
            print(f"!!! [Optimal Solver] ERRORE: Nessuna soluzione ottimale trovata (Status: {pulp.LpStatus[prob.status]}). Verrà usato un piano di fallback (zero azioni).")
            self.action_plan = np.zeros((T, num_cs))

    def get_action(self, env):
        # Se il piano non è stato calcolato, calcolalo ora
        if self.action_plan is None:
            # L'ambiente deve essere in grado di fornire le informazioni future
            # Questo funziona solo se l'ambiente è stato caricato da un replay
            if not hasattr(env, 'get_all_future_ev_sessions'):
                 raise Exception("L'ambiente deve implementare 'get_all_future_ev_sessions' per il solver ottimale.")
            self._solve_once(env)

        current_step = env.current_step
        if current_step < len(self.action_plan):
            return self.action_plan[current_step]
        else:
            # Se la simulazione continua oltre il piano, ritorna zero
            return np.zeros(env.cs)
