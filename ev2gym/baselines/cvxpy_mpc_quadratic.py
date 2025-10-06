import cvxpy as cp
import numpy as np

class OnlineMPC_Solver_Quadratic:
    """
    Risolve il problema di Model Predictive Control (MPC) ONLINE ad ogni step
    utilizzando una funzione obiettivo quadratica con CVXPY.
    Supporta un orizzonte di controllo (Nc) > 1.
    """
    def __init__(self, env, prediction_horizon=5, control_horizon='half',
                 costo_degrado_kwh=0.02, prezzo_ricarica_utente_kwh=0.5, 
                 mpc_desired_soc_factor=0.95, quad_penalty_charge=0.01, 
                 quad_penalty_discharge=0.01, use_adaptive_horizon=False, 
                 h_min=2, h_max=5, lyapunov_alpha=0.1, **kwargs):
        
        self.env = env
        self.costo_degrado_kwh = costo_degrado_kwh
        self.prezzo_ricarica_utente_kwh = prezzo_ricarica_utente_kwh
        self.mpc_desired_soc_factor = mpc_desired_soc_factor
        self.quad_penalty_charge = quad_penalty_charge
        self.quad_penalty_discharge = quad_penalty_discharge

        self.use_adaptive_horizon = use_adaptive_horizon
        if self.use_adaptive_horizon:
            self.h_min = h_min
            self.h_max = h_max
            self.lyapunov_alpha = lyapunov_alpha
            self.current_H = self.h_max # Prediction Horizon (Np)
            print(f"MPC Quadratico Adattivo attivato. H_min={h_min}, H_max={h_max}, current_H={self.current_H}")
        else:
            self.H = prediction_horizon # Prediction Horizon (Np)

        effective_prediction_horizon = self.current_H if self.use_adaptive_horizon else self.H
        if control_horizon == 'half':
            self.Nc = max(1, effective_prediction_horizon // 2)
        else:
            self.Nc = int(control_horizon)
        
        print(f"MPC Quadratico configurato con Np={effective_prediction_horizon}, Nc={self.Nc}")

        self.action_plan = []
        self.plan_step = 0

    def get_action(self, env):
        if self.plan_step < len(self.action_plan):
            action_to_take = self.action_plan[self.plan_step]
            self.plan_step += 1
            return action_to_take

        self.action_plan = []
        self.plan_step = 0

        current_step = env.current_step
        sim_length = env.simulation_length
        num_cs = env.cs
        timescale_h = env.timescale / 60.0
        transformer_limit = env.config['transformer']['max_power']

        if self.use_adaptive_horizon:
            prediction_horizon = min(self.current_H, sim_length - current_step)
        else:
            prediction_horizon = min(self.H, sim_length - current_step)
        
        if prediction_horizon <= 0: return np.zeros(num_cs)

        E_initial = np.zeros(num_cs)
        active_evs = {}
        for i in range(num_cs):
            cs = env.charging_stations[i]
            ev = next((ev for ev in cs.evs_connected if ev is not None), None)
            if ev:
                eta_ch = np.mean(list(ev.charge_efficiency.values())) if isinstance(ev.charge_efficiency, dict) else ev.charge_efficiency
                eta_dis = np.mean(list(ev.discharge_efficiency.values())) if isinstance(ev.discharge_efficiency, dict) else ev.discharge_efficiency
                if eta_dis == 0: eta_dis = 0.9
                active_evs[i] = {'ev': ev, 'eta_ch': eta_ch, 'eta_dis': eta_dis}
                E_initial[i] = ev.get_soc() * ev.battery_capacity

        P_ch = cp.Variable((num_cs, prediction_horizon), nonneg=True)
        P_dis = cp.Variable((num_cs, prediction_horizon), nonneg=True)
        E = cp.Variable((num_cs, prediction_horizon))
        is_charging = cp.Variable((num_cs, prediction_horizon), boolean=True)

        prices_charge = env.charge_prices[0, current_step : current_step + prediction_horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + prediction_horizon]

        linear_profit = cp.sum(
            cp.multiply(self.prezzo_ricarica_utente_kwh - prices_charge - self.costo_degrado_kwh, P_ch) +
            cp.multiply(prices_discharge - self.costo_degrado_kwh, P_dis)
        ) * timescale_h
        
        quadratic_penalty = cp.sum(
            cp.multiply(self.quad_penalty_charge, cp.square(P_ch)) +
            cp.multiply(self.quad_penalty_discharge, cp.square(P_dis))
        ) * timescale_h

        objective = cp.Maximize(linear_profit - quadratic_penalty)

        constraints = []
        for cs_id, data in active_evs.items():
            ev = data['ev']
            eta_ch = data['eta_ch']
            eta_dis = data['eta_dis']
            
            constraints += [P_ch[cs_id, :] <= ev.max_ac_charge_power * is_charging[cs_id, :]]
            constraints += [P_dis[cs_id, :] <= abs(ev.max_discharge_power) * (1 - is_charging[cs_id, :])]
            
            for t in range(prediction_horizon):
                if t == 0:
                    constraints.append(E[cs_id, t] == E_initial[cs_id] + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h)
                else:
                    constraints.append(E[cs_id, t] == E[cs_id, t-1] + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h)

            constraints += [E[cs_id, :] >= ev.min_battery_capacity]
            constraints += [E[cs_id, :] <= ev.battery_capacity]

            departure_step_in_horizon = ev.time_of_departure - current_step - 1
            if 0 <= departure_step_in_horizon < prediction_horizon:
                constraints.append(E[cs_id, departure_step_in_horizon] >= ev.desired_capacity * self.mpc_desired_soc_factor)

        for i in range(num_cs):
            if i not in active_evs:
                constraints += [P_ch[i, :] == 0]
                constraints += [P_dis[i, :] == 0]

        constraints += [cp.sum(P_ch - P_dis, axis=0) <= transformer_limit]

        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.SCIP)
        except cp.error.SolverError:
            print("Solver ECOS_BB non disponibile o fallito. Prova a installare un solver MIQP come SCIP (`pip install pyscipopt`) o usa Gurobi/MOSEK.")
            return np.zeros(num_cs)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            if self.use_adaptive_horizon and active_evs:
                V_current = sum((E_initial[i] - data['ev'].desired_capacity)**2 for i, data in active_evs.items())
                E_next = {}
                for i, data in active_evs.items():
                    p_ch_t0 = P_ch.value[i, 0] if P_ch.value is not None else 0
                    p_dis_t0 = P_dis.value[i, 0] if P_dis.value is not None else 0
                    E_next[i] = E_initial[i] + (p_ch_t0 * data['eta_ch'] - p_dis_t0 / data['eta_dis']) * timescale_h
                V_next = sum((E_next[i] - data['ev'].desired_capacity)**2 for i, data in active_evs.items())

                if V_next <= V_current - self.lyapunov_alpha * V_current:
                    self.current_H = max(self.h_min, self.current_H - 1)
                else:
                    self.current_H = min(self.h_max, self.current_H + 1)

            new_plan = []
            effective_Nc = min(self.Nc, prediction_horizon)
            for t in range(effective_Nc):
                action_t = np.zeros(num_cs)
                for i in range(num_cs):
                    charge = P_ch.value[i, t] if P_ch.value is not None else 0
                    discharge = P_dis.value[i, t] if P_dis.value is not None else 0
                    net_power = (charge or 0) - (discharge or 0)
                    max_power = env.charging_stations[i].get_max_power()
                    if max_power > 0:
                        action_t[i] = net_power / max_power
                new_plan.append(np.clip(action_t, -1, 1))

            self.action_plan = new_plan

            if self.action_plan:
                self.plan_step = 1
                return self.action_plan[0]
            else:
                return np.zeros(num_cs)
        else:
            if self.use_adaptive_horizon:
                self.current_H = min(self.h_max, self.current_H + 1)
            print(f"CVXPY: Ottimizzazione fallita o non ottimale. Status: {problem.status}")
            return np.zeros(num_cs)
