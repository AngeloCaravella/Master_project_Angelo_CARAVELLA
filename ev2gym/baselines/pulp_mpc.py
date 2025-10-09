# --- START OF FILE pulp_mpc.py ---

import pulp
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from pathlib import Path



# =====================================================================================
# --- SOLVER 2: MPC PER V2G PROFIT MAXIMIZATION (come da Paper, Vincoli Rigidi) ---
# =====================================================================================
#V2GProfitMaxMPC_Solver
class OnlineMPC_Solver:
    """
    An MPC solver that maximizes user satisfaction.
    
    STRATEGY:
    1. OBJECTIVE: Maximize user satisfaction by ensuring EVs reach their desired SoC at departure.
    2. V2G ENABLED: The solver can decide to charge or discharge the EV battery to meet the objective.
    3. SOFT CONSTRAINT ON USER SATISFACTION: The solver will try its best to reach the desired SoC.
    """
    def __init__(self, env, prediction_horizon=25, control_horizon='half', 
                 mpc_desired_soc_factor=1,
                 penalty_overload=100.0,
                 **kwargs):
        
        self.env = env
        self.H = prediction_horizon
        self.Nc = max(1, self.H // 2) if control_horizon == 'half' else int(control_horizon)
        self.mpc_desired_soc_factor = mpc_desired_soc_factor
        self.penalty_overload = penalty_overload
        
        print(f"MPC (User Satisfaction) configured with Np={self.H}, Nc={self.Nc}")
        print(f"    -> Target SoC at departure: {self.mpc_desired_soc_factor:.0%}")

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
                eta_dis = np.mean(list(ev.discharge_efficiency.values())) if isinstance(ev.discharge_efficiency, dict) else ev.discharge_efficiency
                active_evs[i] = {'ev': ev, 'eta_ch': eta_ch, 'eta_dis': eta_dis}
                E_initial[i] = ev.get_soc() * ev.battery_capacity

        # --- OBJECTIVE: MAXIMIZE FINAL SOC, with penalty for not meeting desired SoC ---
        prob = pulp.LpProblem(f"V2G_UserSatisfaction_MPC_{current_step}", pulp.LpMaximize)
        
        indices = [(i, j) for i in range(num_cs) for j in range(prediction_horizon)]
        P_ch = pulp.LpVariable.dicts("ChargePower", indices, lowBound=0)
        P_dis = pulp.LpVariable.dicts("DischargePower", indices, lowBound=0)
        b_ch = pulp.LpVariable.dicts("isCharging", indices, cat='Binary')
        b_dis = pulp.LpVariable.dicts("isDischarging", indices, cat='Binary')
        
        E = pulp.LpVariable.dicts("Energy", indices, lowBound=0)
        slack_soc = pulp.LpVariable.dicts("SlackSoC", active_evs.keys(), lowBound=0)
        slack_transformer = pulp.LpVariable.dicts("SlackTransformer", range(prediction_horizon), lowBound=0)

        # Objective is to maximize the sum of final energy levels, with a large penalty for not meeting the desired SoC.
        final_energy_sum = pulp.lpSum(
            E[cs_id, ev_data['ev'].time_of_departure - current_step]
            for cs_id, ev_data in active_evs.items()
            if 0 <= ev_data['ev'].time_of_departure - current_step < prediction_horizon
        )
        soc_penalty = pulp.lpSum(10000 * slack_soc[cs_id] for cs_id in active_evs.keys())
        transformer_overload_penalty = pulp.lpSum(self.penalty_overload * slack_transformer[t] for t in range(prediction_horizon))
        prob.setObjective(final_energy_sum - soc_penalty - transformer_overload_penalty)

        # --- MODEL CONSTRAINTS ---
        for cs_id, data in active_evs.items():
            ev, eta_ch, eta_dis = data['ev'], data['eta_ch'], data['eta_dis']
            
            for t in range(prediction_horizon):
                # Max power constraints and binary logic
                prob += P_ch[cs_id, t] <= ev.max_ac_charge_power * b_ch[cs_id, t]
                prob += P_dis[cs_id, t] <= -ev.max_discharge_power * b_dis[cs_id, t]
                prob += b_ch[cs_id, t] + b_dis[cs_id, t] <= 1
                # Prohibit idle action: must either charge or discharge
                prob += b_ch[cs_id, t] + b_dis[cs_id, t] >= 1
                
                # Battery dynamics
                E_prev = E_initial[cs_id] if t == 0 else E[cs_id, t-1]
                prob += E[cs_id, t] == E_prev + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h
                
                # Battery physical limits
                prob += E[cs_id, t] >= ev.min_battery_capacity
                prob += E[cs_id, t] <= ev.battery_capacity
            
            # --- SOFT CONSTRAINT FOR USER SATISFACTION (aim for 100% SoC) ---
            departure_step_in_horizon = ev.time_of_departure - current_step - 1
            if 0 <= departure_step_in_horizon < prediction_horizon:
                desired_energy = ev.battery_capacity # Aim for 100% SoC
                prob += E[cs_id, departure_step_in_horizon] >= desired_energy - slack_soc[cs_id]

        for i in range(num_cs):
            if i not in active_evs:
                for t in range(prediction_horizon):
                    prob += P_ch[i, t] == 0
                    prob += P_dis[i, t] == 0

        # Hard constraint for transformer limit
        for t in range(prediction_horizon):
            power_evs = pulp.lpSum(P_ch[i, t] - P_dis[i, t] for i in range(num_cs))
            total_power = power_evs + load_forecast[t] + pv_forecast[t]
            limit = transformer_limit_horizon[t] if t < len(transformer_limit_horizon) else transformer_limit_horizon[-1]
            prob += total_power <= limit + slack_transformer[t]

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
                    
                    # Normalize action
                    max_charge_power = env.charging_stations[i].get_max_power()
                    max_discharge_power = env.charging_stations[i].get_min_power() # This is negative

                    if net_power > 0 and max_charge_power > 0:
                        action_t[i] = net_power / max_charge_power
                    elif net_power < 0 and max_discharge_power < 0:
                        action_t[i] = net_power / abs(max_discharge_power)
                    else:
                        action_t[i] = 0

                new_plan.append(np.clip(action_t, -1, 1))
            self.action_plan = new_plan
            if self.action_plan:
                self.plan_step = 1
                return self.action_plan[0]
        else:
            print(f"Warning: MPC problem (but not infeasible) at timestep {current_step}. Status: {pulp.LpStatus[prob.status]}")
            # Fallback: charge as fast as possible
            action_fallback = np.zeros(num_cs)
            for i in active_evs.keys():
                action_fallback[i] = 1.0
            return action_fallback

    
class OptimalOfflineSolver:
    """
    An offline solver that finds the optimal charging schedule to maximize user satisfaction
    for all EVs in the simulation, while respecting all hard constraints.

    This solver serves as the theoretical benchmark for user satisfaction.
    """
    def __init__(self, env, mpc_desired_soc_factor=1, penalty_overload=100.0, **kwargs):
        self.env = env
        self.mpc_desired_soc_factor = mpc_desired_soc_factor
        self.penalty_overload = penalty_overload
        self.action_plan = None
        print(f"Optimal Offline Satisfaction Solver initialized.")

    def _solve_once(self, env):
        print("--- [Optimal Offline Solver] Starting to solve for maximum user satisfaction... ---")
        
        T = env.simulation_length
        num_cs = env.cs
        timescale_h = env.timescale / 60.0
        
        transformer = env.transformers[0]
        load_forecast, pv_forecast = transformer.get_load_pv_forecast(0, T)
        transformer_limit = transformer.get_power_limits(0, T)
        all_ev_sessions = env.get_all_future_ev_sessions()

        # Calculate time to full charge for each EV
        ev_time_to_full = {}
        for ev_id, ev_info in all_ev_sessions.items():
            ev = ev_info['ev']
            energy_needed = ev.battery_capacity - ev.initial_soc_replay * ev.battery_capacity
            
            max_charge_power_kw = ev.max_ac_charge_power 
            
            if max_charge_power_kw > 0:
                time_steps_to_full = energy_needed / (max_charge_power_kw * timescale_h)
                ev_time_to_full[ev_id] = time_steps_to_full
            else:
                ev_time_to_full[ev_id] = float('inf') # Cannot charge

        # =============================================================================
        # --- OBJECTIVE: Maximize the final state of charge for all EVs (soft constraint) ---
        # =============================================================================
        prob = pulp.LpProblem("Maximize_Final_SoC_Soft", pulp.LpMaximize)
        
        # Definiamo le variabili come prima
        indices = [(i, t) for i in range(num_cs) for t in range(T)]
        P_ch = pulp.LpVariable.dicts("ChargePower", indices, lowBound=0)
        P_dis = pulp.LpVariable.dicts("DischargePower", indices, lowBound=0)
        b_ch = pulp.LpVariable.dicts("isCharging", indices, cat='Binary')
        b_dis = pulp.LpVariable.dicts("isDischarging", indices, cat='Binary')
        ev_indices = [(ev_id, t) for ev_id in all_ev_sessions.keys() for t in range(T)]
        E = pulp.LpVariable.dicts("Energy", ev_indices, lowBound=0)
        slack_soc = pulp.LpVariable.dicts("SlackSoC", all_ev_sessions.keys(), lowBound=0) # Slack for user satisfaction
        slack_transformer = pulp.LpVariable.dicts("SlackTransformer", range(T), lowBound=0)

        # The objective is to maximize the sum of energy for all EVs at their departure time,
        # with a large penalty for not meeting the desired SoC.
        final_energy_sum = pulp.lpSum(
            E[ev_id, ev_info['departure_step'] - 1]
            for ev_id, ev_info in all_ev_sessions.items() if ev_info['departure_step'] > ev_info['arrival_step']
        )
        soc_penalty = pulp.lpSum(10000 * slack_soc[ev_id] for ev_id in all_ev_sessions.keys())
        transformer_overload_penalty = pulp.lpSum(self.penalty_overload * slack_transformer[t] for t in range(T))
        prob.setObjective(final_energy_sum - soc_penalty - transformer_overload_penalty)
        
        # =============================================================================
        # --- THE CONSTRAINTS ARE NOW ALL HARD AND ABSOLUTE ---
        # =============================================================================
        
        # Vincoli per ogni sessione EV
        for ev_id, ev_info in all_ev_sessions.items():
            ev = ev_info['ev']
            cs_id = ev_info['cs_id']
            t_arrival = ev_info['arrival_step']
            t_departure = ev_info['departure_step']
            
            eta_ch = np.mean(list(ev.charge_efficiency.values()))
            eta_dis = np.mean(list(ev.discharge_efficiency.values()))
            initial_energy_kwh = ev.initial_soc_replay * ev.battery_capacity

            for t in range(t_arrival, t_departure):
                E_prev = initial_energy_kwh if t == t_arrival else E[ev_id, t-1]
                prob += E[ev_id, t] == E_prev + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h
                prob += E[ev_id, t] <= ev.battery_capacity
                prob += E[ev_id, t] >= ev.min_battery_capacity

            # SOFT CONSTRAINT FOR USER SATISFACTION (aim for 100% SoC)
            desired_energy = ev.battery_capacity # Always aim for 100% SoC for OptimalOfflineSolver
            if t_departure > t_arrival:
                prob += E[ev_id, t_departure - 1] >= desired_energy - slack_soc[ev_id]

            # V2G is implicitly discouraged by maximizing final SoC and penalizing transformer overload.
            # No explicit conditional V2G constraint needed here.

        # Vincoli di potenza delle stazioni
        for i, t in indices:
            current_ev_id = None
            for ev_id, ev_info in all_ev_sessions.items():
                if ev_info['cs_id'] == i and ev_info['arrival_step'] <= t < ev_info['departure_step']:
                    current_ev_id = ev_id
                    break
            if current_ev_id:
                ev = all_ev_sessions[current_ev_id]['ev']
                prob += P_ch[i, t] <= ev.max_ac_charge_power * b_ch[i, t]
                prob += P_dis[i, t] <= -ev.max_discharge_power * b_dis[i, t]
                prob += b_ch[i, t] + b_dis[i, t] <= 1
            else:
                prob += P_ch[i, t] == 0
                prob += P_dis[i, t] == 0

        # HARD AND INFALLIBLE TRANSFORMER CONSTRAINT (without slack)
        for t in range(T):
            power_evs = pulp.lpSum(P_ch[i, t] - P_dis[i, t] for i in range(num_cs))
            total_power = power_evs + load_forecast[t] - pv_forecast[t]
            prob += total_power <= transformer_limit[t] + slack_transformer[t]

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            print("--- [Optimal Offline Solver] SUCCESS: Optimal charging plan found. ---")
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
            print(f"--- [Optimal Offline Solver] FAILURE: Problem is infeasible (Status: {pulp.LpStatus[prob.status]}). ---")
            self.action_plan = np.zeros((T, num_cs))
        
    def get_action(self, env):
        if self.action_plan is None:
            if not hasattr(env, 'get_all_future_ev_sessions'):
                 raise Exception("The environment must implement 'get_all_future_ev_sessions' for the optimal solver.")
            self._solve_once(env)
        return self.action_plan[env.current_step] if env.current_step < len(self.action_plan) else np.zeros(env.cs)
