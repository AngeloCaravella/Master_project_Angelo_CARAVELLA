import math
import numpy as np
from collections import deque
from copy import deepcopy

# ===============================================================================================
# ========= NUOVA FUNZIONE ADATTIVA, VELOCE E OTTIMIZZATA PER IL PROFITTO =========================
# ===============================================================================================

def FastProfitAdaptiveReward(env, total_costs, user_satisfaction_list, *args):
    """
    Versione migliorata: clamp dei bonus, penalità relative alla scala di total_costs,
    e tracciamento più chiaro dei componenti di reward.
    """
    # Inizializza solo se necessario
    if not hasattr(env, 'satisfaction_history'):
        env.satisfaction_history = deque([1.0], maxlen=100)
        env.overload_frequency = deque([0], maxlen=100)

    # --- Pre-elaborazione: conversione a numpy per velocità ---
    sat_hist = np.fromiter(env.satisfaction_history, dtype=float)
    ov_hist = np.fromiter(env.overload_frequency, dtype=float)
    avg_satisfaction = float(sat_hist.mean()) if sat_hist.size else 1.0
    overload_freq = float(ov_hist.mean()) if ov_hist.size else 0.0

    # --- Profitto dinamico (bonus clampati in [0,1]) ---
    satisfaction_bonus = (avg_satisfaction - 0.8) * 5.0
    satisfaction_bonus = float(np.clip(satisfaction_bonus, 0.0, 1.0))

    overload_bonus = 1.0 - overload_freq * 5.0
    overload_bonus = float(np.clip(overload_bonus, 0.0, 1.0))

    profit_multiplier = 0.1 + 0.9 * (satisfaction_bonus * overload_bonus)  # in [0.1, 1.0]
    base_profit = float(total_costs) * profit_multiplier

    reward = base_profit
    reward_components = {
        'base_profit': base_profit,
        'profit_multiplier': profit_multiplier,
        'satisfaction_bonus': satisfaction_bonus,
        'overload_bonus': overload_bonus
    }

    # --- Penalità soddisfazione (relativa alla scala dei costi) ---
    if user_satisfaction_list:
        user_satisfaction_arr = np.fromiter(user_satisfaction_list, dtype=float)
        min_satisfaction = float(user_satisfaction_arr.min())
        if min_satisfaction < 0.95:
            # alpha tunabile: imposta in base a quanto vuoi penalizzare (0.5..2.0)
            alpha = 1.0
            scale = max(1.0, abs(total_costs))
            penalty = - alpha * scale * (1 - avg_satisfaction)**2 * (1 - min_satisfaction)
            reward += penalty
            reward_components['satisfaction_penalty'] = penalty

    # --- Penalità sovraccarico dei trasformatori (log1p e relativa alla scala dei costi) ---
    overload_values = np.fromiter(
        (tr.get_how_overloaded() for tr in env.transformers),
        dtype=float
    ) if getattr(env, 'transformers', None) is not None else np.array([], dtype=float)

    current_overload_amount = float(overload_values.sum()) if overload_values.size else 0.0
    if current_overload_amount > 0.0:
        beta = 0.5  # tunabile: quanto pesare l'overload rispetto al profitto
        scale = max(1.0, abs(total_costs))
        overload_penalty = -5.0 * beta - (10.0 * beta * overload_freq * np.log1p(current_overload_amount)) 
        # rendi la penalty proporzionale alla scala dei costi (opzionale)
        overload_penalty *= (scale / (scale + 10.0))
        reward += overload_penalty
        reward_components['transformer_penalty'] = overload_penalty
        reward_components['current_overload_amount'] = current_overload_amount

    # --- Aggiornamento storico ---
    env.satisfaction_history.append(
        float(np.mean(user_satisfaction_list)) if user_satisfaction_list else 1.0
    )
    env.overload_frequency.append(1 if current_overload_amount > 0.0 else 0)

    # --- Tracciamento opzionale ---
    if hasattr(env, 'step_info'):
        env.step_info['reward_components'] = reward_components

    return float(reward)

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    for score in user_satisfaction_list:
        reward -= 100 * math.exp(-10 * score)
    return reward
