# --- INIZIO DEL FILE pulp_mpc.py ---

import numpy as np
import pulp

class OnlineMPC_Solver:
    """
    Risolve il problema MPC implementando STRETTAMENTE la formulazione matematica
    descritta nel paper di riferimento (equazioni 10-24).
    Utilizza il risolutore di default CBC.
    """

    def __init__(self, env, prediction_horizon=10, control_horizon='half',
                 use_adaptive_horizon=False, h_min=2, h_max=5, lyapunov_alpha=0.1,
                 **kwargs):
        self.env = env

        # --- Logica per Orizzonte Adattivo basato su Lyapunov ---
        self.use_adaptive_horizon = use_adaptive_horizon
        if self.use_adaptive_horizon:
            self.h_min = h_min
            self.h_max = h_max
            self.lyapunov_alpha = lyapunov_alpha
            self.current_H = self.h_max # Inizializza con l'orizzonte massimo
        else:
            self.H = prediction_horizon

        # --- Impostazione dell'orizzonte di controllo (Nc) ---
        effective_prediction_horizon = self.current_H if self.use_adaptive_horizon else self.H
        if control_horizon == 'half':
            self.Nc = max(1, effective_prediction_horizon // 2)
        else:
            self.Nc = int(control_horizon)
        
        print(f"MPC (Strict Paper Formulation, CBC) configurato con Np={effective_prediction_horizon}, Nc={self.Nc}")
        print(f"    -> V2G Sempre Attivo, Soddisfazione Utente: Vincolo Rigido")

    def get_action(self, env):
        current_step = env.current_step
        sim_length = env.simulation_length
        num_cs = env.cs
        timescale_h = env.timescale / 60.0
        
        prediction_horizon = min(self.current_H if self.use_adaptive_horizon else self.H, sim_length - current_step)
        if prediction_horizon <= 0:
            return np.zeros(num_cs)

        # --- Parametri del problema ---
        transformer = env.transformers[0]
        # Limite di potenza del trasformatore (P_tr_bar)
        transformer_limit_horizon = transformer.get_power_limits(current_step, prediction_horizon)
        # Carico inflessibile (P_L) e produzione PV (P_PV)
        load_forecast, pv_forecast = transformer.get_load_pv_forecast(current_step, prediction_horizon)
        inflexible_load_horizon = load_forecast
        pv_generation_horizon = pv_forecast
        
        # Prezzi di carica (c_ch) e scarica (c_dis)
        prices_charge = env.charge_prices[0, current_step : current_step + prediction_horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + prediction_horizon]

        # Dizionario per memorizzare i dati degli EV attivi
        active_evs = {}
        for i in range(num_cs):
            cs = env.charging_stations[i]
            ev = next((ev for ev in cs.evs_connected if ev is not None), None)
            if ev:
                # Assumiamo valori standard per tensione e fattore di potenza se non disponibili
                V = getattr(cs, 'voltage', 230.0) / 1000.0 # in kV per coerenza con kW
                phi = getattr(cs, 'power_factor', 1.0)
                
                # Calcolo delle efficienze medie
                eta_ch = np.mean(list(ev.charge_efficiency.values())) if isinstance(ev.charge_efficiency, dict) else ev.charge_efficiency
                eta_dis = np.mean(list(ev.discharge_efficiency.values())) if isinstance(ev.discharge_efficiency, dict) else ev.discharge_efficiency
                if eta_dis == 0: eta_dis = 0.9 # Evita divisione per zero

                active_evs[i] = {
                    'ev': ev,
                    'eta_ch': eta_ch,
                    'eta_dis': eta_dis,
                    'V': V,
                    'phi': phi,
                    'E_initial': ev.get_soc() * ev.battery_capacity,
                    'I_ch_bar': ev.max_ac_charge_power / V, # Corrente massima di carica
                    'I_dis_bar': abs(ev.max_discharge_power) / V, # Corrente massima di scarica
                }

        # --- Definizione del problema di ottimizzazione ---
        prob = pulp.LpProblem(f"Strict_MPC_Profit_Maximization_{current_step}", pulp.LpMaximize)
        
        # --- Variabili decisionali ---
        # j, i sono collassati in un unico indice `i` (un EV per stazione di ricarica)
        # t è l'indice temporale
        indices = [(i, t) for i in active_evs.keys() for t in range(prediction_horizon)]
        
        # Correnti di carica e scarica
        I_ch = pulp.LpVariable.dicts("ChargeCurrent", indices, lowBound=0)
        I_dis = pulp.LpVariable.dicts("DischargeCurrent", indices, lowBound=0)
        
        # Variabili binarie per lo stato di carica/scarica
        omega_ch = pulp.LpVariable.dicts("IsCharging", indices, cat='Binary')
        omega_dis = pulp.LpVariable.dicts("IsDischarging", indices, cat='Binary')
        
        # Energia della batteria
        E = pulp.LpVariable.dicts("Energy", indices, lowBound=0)

        # --- Espressioni per la Potenza (Equazioni 10, 11) ---
        # NOTA: Le equazioni (10) e (11) sono non-lineari (I_ch * omega_ch).
        # Per mantenerle lineari, leghiamo la corrente alla variabile binaria
        # usando il vincolo I_ch <= I_ch_bar * omega_ch (vedi vincolo 15).
        # La potenza diventa quindi un'espressione lineare dipendente solo da I_ch/I_dis.
        
        # (10) Potenza di carica P_ch
        P_ch = { (i, t): I_ch[i, t] * active_evs[i]['V'] * np.sqrt(active_evs[i]['phi']) * active_evs[i]['eta_ch'] 
                 for i, t in indices }

        # (11) Potenza di scarica P_dis
        # NOTA: L'efficienza di scarica (eta_dis) dovrebbe dividere la potenza, non moltiplicarla.
        # Il paper la usa come moltiplicatore, e noi seguiamo strettamente quella formulazione.
        P_dis = { (i, t): I_dis[i, t] * active_evs[i]['V'] * np.sqrt(active_evs[i]['phi']) * active_evs[i]['eta_dis']
                  for i, t in indices }

        # --- Funzione Obiettivo (Equazione 23) ---
        # max Σ (-P_ch * c_ch + P_dis * c_dis) * Δt
        objective = pulp.lpSum(
            (-P_ch[i, t] * prices_charge[t] + P_dis[i, t] * prices_discharge[t]) * timescale_h
            for i, t in indices
        )
        prob.setObjective(objective)

        # --- Vincoli ---
        for cs_id in active_evs.keys():
            ev = active_evs[cs_id]['ev']
            dep_step = ev.time_of_departure - current_step - 1

            for t in range(prediction_horizon):
                # (21) Vincolo di esclusione: non si può caricare e scaricare simultaneamente
                prob += omega_ch[cs_id, t] + omega_dis[cs_id, t] <= 1

                # (15) Limite superiore sulla corrente di carica
                # Questo vincolo lega anche la corrente alla variabile binaria omega_ch
                prob += I_ch[cs_id, t] <= active_evs[cs_id]['I_ch_bar'] * omega_ch[cs_id, t]

                # (16) Limite sulla corrente di scarica
                # NOTA: La formula I_dis >= I_dis_bar è probabilmente un errore di battitura.
                # Implementiamo il vincolo più plausibile I_dis <= I_dis_bar.
                prob += I_dis[cs_id, t] <= active_evs[cs_id]['I_dis_bar'] * omega_dis[cs_id, t]

                # (13) Dinamica della batteria
                # NOTA: La formula E_t = E_{t-1} + (P_ch + P_dis) * dt è fisicamente errata.
                # La potenza di scarica (P_dis) dovrebbe essere sottratta.
                # Seguiamo strettamente la formula del paper.
                E_prev = active_evs[cs_id]['E_initial'] if t == 0 else E[cs_id, t-1]
                prob += E[cs_id, t] == E_prev + (P_ch[cs_id, t] + P_dis[cs_id, t]) * timescale_h
                
                # (12) Limite superiore di capacità della batteria
                prob += E[cs_id, t] <= ev.battery_capacity
                # Aggiungiamo un limite inferiore implicito per evitare che la batteria si scarichi troppo
                prob += E[cs_id, t] >= ev.min_battery_capacity

            # (24) Vincolo sull'energia desiderata alla partenza
            if 0 <= dep_step < prediction_horizon:
                desired_energy = ev.desired_capacity
                prob += E[cs_id, dep_step] >= desired_energy

        for t in range(prediction_horizon):
            # (19) Potenza totale assorbita/erogata dagli EV
            # NOTA: Anche qui, P_dis dovrebbe essere sottratta. Seguiamo la formula.
            P_EVs_t = pulp.lpSum(P_ch.get((i, t), 0) + P_dis.get((i, t), 0) for i in active_evs.keys())
            
            # (20) Limite di potenza del trasformatore (vincolo rigido)
            # P_EVs + P_L + P_PV <= P_tr_bar - P_DR
            # Assumiamo P_DR (Demand Response) = 0 se non specificato
            transformer_limit = transformer_limit_horizon[t]
            inflexible_load = inflexible_load_horizon[t]
            pv_generation = pv_generation_horizon[t]
            
            prob += P_EVs_t + inflexible_load + pv_generation <= transformer_limit

        # Vincoli (17) e (18) sulla corrente totale della stazione di ricarica
        # sono omessi perché il modello opera a livello di potenza aggregata sul trasformatore,
        # e i limiti di corrente dei singoli EV sono già inclusi.

        # --- Risoluzione del problema ---
        solver = pulp.PULP_CBC_CMD(
            options=['logLevel', '1']
        )


        prob.solve(solver)
        status = pulp.LpStatus[prob.status]


        if status == 'Optimal':
            action = np.zeros(num_cs)
            for i in active_evs.keys():
                # Calcoliamo la potenza netta per il primo step (t=0)
                charge_power = pulp.value(P_ch.get((i, 0), 0))
                discharge_power = pulp.value(P_dis.get((i, 0), 0))
                
                # L'azione è la potenza netta (carica > 0, scarica < 0)
                # NOTA: A causa dei probabili errori nelle equazioni (13) e (19),
                # il risolutore potrebbe non trovare una soluzione ottimale sensata.
                # Per l'azione, usiamo la definizione fisica corretta: carica - scarica.
                net_power = (charge_power or 0) - (discharge_power or 0)
                
                max_power = env.charging_stations[i].get_max_power()
                if max_power > 0:
                    action[i] = net_power / max_power
            
            return np.clip(action, -1, 1)
        
        elif status == 'Infeasible':
            print(f"\n--- MPC WARNING (Strict): PROBLEMA INFATTIBILE (Step: {current_step}) ---\n")
        else:
            print(f"\n--- MPC WARNING (Strict): Stato non ottimale: {status} (Step: {current_step}) ---\n")

        return np.zeros(num_cs)

# --- FINE DEL FILE pulp_mpc.py ---
