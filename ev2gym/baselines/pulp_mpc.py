# --- START OF FILE pulp_mpc.py ---

import pulp
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from pathlib import Path

class OnlineMPC_Solver:
    """
    Risolve il problema di Model Predictive Control (MPC) ONLINE.
    Questa versione è configurata per dare alta priorità alla soddisfazione dell'utente,
    utilizzando vincoli morbidi e pesi configurabili per bilanciare profitto e penalità,
    in modo simile alla funzione di reward di un agente RL.
    """

    def __init__(self, env, prediction_horizon=5, control_horizon='half', 
                 mpc_desired_soc_factor=0.95, use_adaptive_horizon=False, 
                 h_min=2, h_max=5, lyapunov_alpha=0.1,
                 # --- PARAMETRI DI BILANCIAMENTO PER L'OBIETTIVO ---
                 profit_weight=0.1,              # Riduce l'importanza del profitto
                 penalty_overload=100.0,           # Penalità per ogni kW di sovraccarico del trasformatore
                 penalty_user_satisfaction=1000.0, # Penalità elevata per ogni kWh mancante all'utente
                 **kwargs):
        
        self.env = env
        self.mpc_desired_soc_factor = mpc_desired_soc_factor

        # --- ATTRIBUTI PER IL BILANCIAMENTO DELLA FUNZIONE OBIETTIVO ---
        self.profit_weight = profit_weight
        self.penalty_overload = penalty_overload
        self.penalty_user_satisfaction = penalty_user_satisfaction

        # --- Logica per Orizzonte Adattivo basato su Lyapunov ---
        self.use_adaptive_horizon = use_adaptive_horizon
        if self.use_adaptive_horizon:
            self.h_min = h_min
            self.h_max = h_max
            self.lyapunov_alpha = lyapunov_alpha
            self.current_H = self.h_max
        else:
            self.H = prediction_horizon

        # --- Impostazione dell'orizzonte di controllo (Nc) ---
        effective_prediction_horizon = self.current_H if self.use_adaptive_horizon else self.H
        if control_horizon == 'half':
            self.Nc = max(1, effective_prediction_horizon // 2)
        else:
            self.Nc = int(control_horizon)
        
        print(f"MPC (User Satisfaction) configurato con Np={effective_prediction_horizon}, Nc={self.Nc}")
        print(f"    -> Pesi Obiettivo: Profitto={self.profit_weight}, Penalità Sovraccarico={self.penalty_overload}, Penalità Soddisfazione={self.penalty_user_satisfaction}")

        # --- Cache per il piano di azioni ---
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
        prediction_horizon = min(self.current_H if self.use_adaptive_horizon else self.H, sim_length - current_step)
        if prediction_horizon <= 0: return np.zeros(num_cs)

        transformer = env.transformers[0]
        transformer_limit_horizon = transformer.get_power_limits(current_step, prediction_horizon)

        E_initial, active_evs, degradation_costs, user_prices = np.zeros(num_cs), {}, {}, {}
        for i in range(num_cs):
            cs = env.charging_stations[i]
            ev = next((ev for ev in cs.evs_connected if ev is not None), None)
            if ev:
                eta_ch = np.mean(list(ev.charge_efficiency.values())) if isinstance(ev.charge_efficiency, dict) else ev.charge_efficiency
                eta_dis = np.mean(list(ev.discharge_efficiency.values())) if isinstance(ev.discharge_efficiency, dict) else ev.discharge_efficiency
                if eta_dis == 0: eta_dis = 0.9
                active_evs[i] = {'ev': ev, 'eta_ch': eta_ch, 'eta_dis': eta_dis}
                E_initial[i] = ev.get_soc() * ev.battery_capacity
                degradation_costs[i] = getattr(ev, 'costo_degrado_kwh', 0.02)
                user_prices[i] = getattr(cs, 'prezzo_ricarica_utente_kwh', 0.5)

        prob = pulp.LpProblem(f"Online_MPC_UserSat_{current_step}", pulp.LpMaximize)
        indices = [(i, j) for i in range(num_cs) for j in range(prediction_horizon)]
        P_ch = pulp.LpVariable.dicts("ChargePower", indices, lowBound=0)
        P_dis = pulp.LpVariable.dicts("DischargePower", indices, lowBound=0)
        is_charging = pulp.LpVariable.dicts("IsCharging", indices, cat='Binary')
        E = pulp.LpVariable.dicts("Energy", indices, lowBound=0)
        slack_overload = pulp.LpVariable.dicts("SlackOverload", range(prediction_horizon), lowBound=0)
        slack_soc_deficit = pulp.LpVariable.dicts("SlackSoCDeficit", list(active_evs.keys()), lowBound=0)

        prices_charge = env.charge_prices[0, current_step : current_step + prediction_horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + prediction_horizon]
        
        profit_objective = pulp.lpSum(
            ((user_prices.get(i, 0.5) - prices_charge[t] - degradation_costs.get(i, 0.02)) * P_ch[i, t] +
             (prices_discharge[t] - degradation_costs.get(i, 0.02)) * P_dis[i, t]) * timescale_h
            for i in active_evs.keys() for t in range(prediction_horizon)
        )
        overload_penalty = pulp.lpSum(self.penalty_overload * slack_overload[t] for t in range(prediction_horizon))
        satisfaction_penalty = pulp.lpSum(self.penalty_user_satisfaction * slack_soc_deficit[i] for i in active_evs.keys())
        prob.setObjective(self.profit_weight * profit_objective - overload_penalty - satisfaction_penalty)

        for cs_id, data in active_evs.items():
            ev, eta_ch, eta_dis = data['ev'], data['eta_ch'], data['eta_dis']
            for t in range(prediction_horizon):
                prob += P_ch[cs_id, t] <= ev.max_ac_charge_power * is_charging[cs_id, t]
                prob += P_dis[cs_id, t] <= abs(ev.max_discharge_power) * (1 - is_charging[cs_id, t])
                E_prev = E_initial[cs_id] if t == 0 else E[cs_id, t-1]
                prob += E[cs_id, t] == E_prev + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h
                prob += E[cs_id, t] >= ev.min_battery_capacity
                prob += E[cs_id, t] <= ev.battery_capacity
            
            dep_step = ev.time_of_departure - current_step - 1
            if 0 <= dep_step < prediction_horizon:
                desired_energy = ev.desired_capacity * self.mpc_desired_soc_factor
                prob += E[cs_id, dep_step] >= desired_energy - slack_soc_deficit[cs_id]

        for i in range(num_cs):
            if i not in active_evs:
                for t in range(prediction_horizon):
                    prob += P_ch[i, t] == 0
                    prob += P_dis[i, t] == 0

        for t in range(prediction_horizon):
            limit = transformer_limit_horizon[t] if t < len(transformer_limit_horizon) else transformer_limit_horizon[-1]
            prob += pulp.lpSum(P_ch[i, t] - P_dis[i, t] for i in range(num_cs)) <= limit + slack_overload[t]

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            new_plan = []
            effective_Nc = min(self.Nc, prediction_horizon)
            for t in range(effective_Nc):
                action_t = np.zeros(num_cs)
                for i in range(num_cs):
                    charge = pulp.value(P_ch[i, t])
                    discharge = pulp.value(P_dis[i, t])
                    net_power = (charge or 0) - (discharge or 0)
                    max_power = env.charging_stations[i].get_max_power()
                    if max_power > 0: action_t[i] = net_power / max_power
                new_plan.append(np.clip(action_t, -1, 1))
            self.action_plan = new_plan
            if self.action_plan:
                self.plan_step = 1
                return self.action_plan[0]
        return np.zeros(num_cs)
