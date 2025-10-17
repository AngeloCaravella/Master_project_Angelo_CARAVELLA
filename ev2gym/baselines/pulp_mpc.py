import numpy as np
import pulp

class OnlineMPC_Solver:
    """
    Risolve il problema MPC implementando la formulazione matematica
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
                # Assumiamo valori standard per tensione e fattore di potenza
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
                    #'I_ch_bar': ev.max_ac_charge_power / V, # Corrente massima di carica
                    #'I_dis_bar': abs(ev.max_discharge_power) / V, # Corrente massima di scarica
		    'I_ch_bar': 32,
		    'I_dis_bar': 32
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



        P_ch = { (i, t): I_ch[i, t] * active_evs[i]['V'] * np.sqrt(active_evs[i]['phi']) 
                 for i, t in indices }

        P_dis = { (i, t): I_dis[i, t] * active_evs[i]['V'] * np.sqrt(active_evs[i]['phi'])
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
             
                prob += I_dis[cs_id, t] <= active_evs[cs_id]['I_dis_bar'] * omega_dis[cs_id, t]
            

                # (13) Dinamica della batteria
              
                E_prev = active_evs[cs_id]['E_initial'] if t == 0 else E[cs_id, t-1]
                prob += E[cs_id, t] == E_prev + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h
                
                # (12) Limite superiore di capacità della batteria
                prob += E[cs_id, t] <= ev.battery_capacity
                # Limite inferiore per evitare scariche eccessive
                prob += E[cs_id, t] >= ev.min_battery_capacity

            
            # (24) Vincolo sull'energia desiderata alla partenza
            if 0 <= dep_step < prediction_horizon:
                desired_energy = ev.desired_capacity
                prob += E[cs_id, dep_step] >= desired_energy

        for t in range(prediction_horizon):
            # (19) Potenza totale assorbita/erogata dagli EV
      
            P_EVs_t = pulp.lpSum(P_ch.get((i, t), 0) + P_dis.get((i, t), 0) for i in active_evs.keys())
            
            # (20) Limite di potenza del trasformatore (vincolo rigido)
           
            transformer_limit = transformer_limit_horizon[t]
            inflexible_load = inflexible_load_horizon[t]
            pv_generation = pv_generation_horizon[t]
            
            prob += P_EVs_t + inflexible_load + pv_generation <= transformer_limit

   
        

        V_current = 0
        if self.use_adaptive_horizon:
            for cs_id in active_evs.keys():
                ev = active_evs[cs_id]['ev']
                if current_step < ev.time_of_departure:
                    E_current = active_evs[cs_id]['E_initial']
                    E_des = ev.desired_capacity
                    V_current += (E_current - E_des)**2

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        status = pulp.LpStatus[prob.status]

        if self.use_adaptive_horizon:
            if status == 'Optimal':
                V_next = 0
                for cs_id in active_evs.keys():
                    ev = active_evs[cs_id]['ev']
                    if current_step < ev.time_of_departure:
                        E_next_planned = pulp.value(E[cs_id, 0])
                        E_des = ev.desired_capacity
                        V_next += (E_next_planned - E_des)**2
                
                # Lyapunov stability condition from algorithm
                if V_next <= V_current - self.lyapunov_alpha * V_current:
                    # System is converging, decrease horizon
                    self.current_H = max(self.h_min, self.current_H - 1)
                else:
                    # System is not converging fast enough, increase horizon
                    self.current_H = min(self.h_max, self.current_H + 1)
            else:
                # Solver failed, increase horizon as a safeguard
                self.current_H = min(self.h_max, self.current_H + 1)


        if status == 'Optimal':
            action = np.zeros(num_cs)
            for i in active_evs.keys():
                # Calcoliamo la potenza netta per il primo step (t=0)
                charge_power = pulp.value(P_ch.get((i, 0), 0))
                discharge_power = pulp.value(P_dis.get((i, 0), 0))
               
                # L'azione è la potenza netta (carica > 0, scarica < 0)
                # NOTA: A causa dei probabili errori nelle equazioni (13) e (19),
            
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
