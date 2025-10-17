import math
import numpy as np
from collections import deque
from copy import deepcopy

# ===============================================================================================
# ========= NUOVA FUNZIONE ADATTIVA, VELOCE E OTTIMIZZATA PER IL PROFITTO =========================
# ===============================================================================================

def FastProfitAdaptiveReward(env, total_costs, user_satisfaction_list, *args):
    """
    Versione ottimizzata per la velocità della funzione adattiva di reward.
    """
    # Inizializza solo se necessario
    if not hasattr(env, 'satisfaction_history'):
        env.satisfaction_history = deque([1.0], maxlen=100)
        env.overload_frequency = deque([0], maxlen=100)

    # --- Pre-elaborazione: conversione a numpy per velocità ---
    sat_hist = np.fromiter(env.satisfaction_history, dtype=float)
    ov_hist = np.fromiter(env.overload_frequency, dtype=float)
    avg_satisfaction = sat_hist.mean() if sat_hist.size else 1.0
    overload_freq = ov_hist.mean() if ov_hist.size else 0.0

    # --- Profitto dinamico ---
    satisfaction_bonus = max(0.0, (avg_satisfaction - 0.9) * 10.0)  # evita divisione
    overload_bonus = max(0.0, 1.0 - overload_freq * 5.0)            # evita divisione
    profit_multiplier = 0.1 + 0.9 * satisfaction_bonus * overload_bonus

    reward = total_costs * profit_multiplier
    reward_components = {'profit': reward, 'profit_multiplier': profit_multiplier}

    # --- Penalità soddisfazione ---
    if user_satisfaction_list:
        user_satisfaction_arr = np.fromiter(user_satisfaction_list, dtype=float)
        min_satisfaction = user_satisfaction_arr.min()
        if min_satisfaction < 0.95:
            penalty = -200.0 * (1 - avg_satisfaction)**2 * (1 - min_satisfaction)
            reward += penalty
            reward_components['satisfaction_penalty'] = penalty

    # --- Penalità sovraccarico ---
    overload_values = np.fromiter(
        (tr.get_how_overloaded() for tr in env.transformers),
        dtype=float
    )
    current_overload_amount = overload_values.sum()
    if current_overload_amount > 0.0:
        overload_penalty = -5.0 - (50.0 * overload_freq * current_overload_amount)
        reward += overload_penalty
        reward_components['transformer_penalty'] = overload_penalty

    # --- Aggiornamento storico ---
    env.satisfaction_history.append(
        float(np.mean(user_satisfaction_list)) if user_satisfaction_list else 1.0
    )
    env.overload_frequency.append(1 if current_overload_amount > 0.0 else 0)

    # --- Tracciamento opzionale ---
    if hasattr(env, 'step_info'):
        env.step_info['reward_components'] = reward_components

    return reward


def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    for score in user_satisfaction_list:
        reward -= 100 * math.exp(-10 * score)
    return reward
