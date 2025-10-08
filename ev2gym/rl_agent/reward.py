import math
import numpy as np
from collections import deque
from copy import deepcopy

# ===============================================================================================
# ========= NUOVA FUNZIONE ADATTIVA, VELOCE E OTTIMIZZATA PER IL PROFITTO =========================
# ===============================================================================================

def FastProfitAdaptiveReward(env, total_costs, user_satisfaction_list, *args):
    """
    Una funzione di reward adattiva, ottimizzata per la velocità computazionale e la massimizzazione del profitto.

    Privilegia in modo aggressivo il profitto economico, applicando penalità che si adattano
    dinamicamente in base alla performance recente del sistema riguardo la soddisfazione dell'utente
    e il sovraccarico del trasformatore.
    """
    # Inizializza i tracker di stato se non esistono. Per una maggiore pulizia,
    # questa logica andrebbe nella funzione __init__ o reset dell'ambiente.
    if not hasattr(env, 'satisfaction_history'):
        env.satisfaction_history = deque(maxlen=100)
    if not hasattr(env, 'overload_frequency'):
        env.overload_frequency = deque(maxlen=100)

    # La ricompensa principale è il profitto economico diretto.
    reward = total_costs
    reward_components = {'profit': reward}

    # Penalità Adattiva per la Soddisfazione dell'Utente
    avg_satisfaction = sum(env.satisfaction_history) / len(env.satisfaction_history) if env.satisfaction_history else 1.0
    satisfaction_severity_multiplier = 50.0 * (1 - avg_satisfaction)**2
    
    current_satisfaction_penalty = 0
    if user_satisfaction_list:
        min_satisfaction = min(user_satisfaction_list)
        if min_satisfaction < 0.95:
            current_satisfaction_penalty = -satisfaction_severity_multiplier * (1 - min_satisfaction)
    
    if current_satisfaction_penalty < 0:
        reward += current_satisfaction_penalty
        reward_components['satisfaction_penalty'] = current_satisfaction_penalty

    # Penalità Adattiva per il Sovraccarico del Trasformatore
    overload_freq = sum(env.overload_frequency) / len(env.overload_frequency) if env.overload_frequency else 0.0
    overload_severity_multiplier = 50.0 * overload_freq

    current_overload_amount = sum(tr.get_how_overloaded() for tr in env.transformers)
    if current_overload_amount > 0:
        overload_penalty = -5.0 - (overload_severity_multiplier * current_overload_amount)
        reward += overload_penalty
        reward_components['transformer_penalty'] = overload_penalty

    # Aggiorna le cronologie per il prossimo passo
    avg_current_satisfaction = sum(user_satisfaction_list) / len(user_satisfaction_list) if user_satisfaction_list else 1.0
    env.satisfaction_history.append(avg_current_satisfaction)
    env.overload_frequency.append(1 if current_overload_amount > 0 else 0)

    if hasattr(env, 'step_info'):
        env.step_info['reward_components'] = reward_components

    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    for score in user_satisfaction_list:
        reward -= 500 * math.exp(-10 * score)
    return reward
