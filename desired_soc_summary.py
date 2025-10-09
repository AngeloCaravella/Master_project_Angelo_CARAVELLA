import pandas as pd

def print_desired_soc_summary(env):
    """
    Prints a summary of the desired SoC for the different EV distributions.
    """
    if not env.config.get('heterogeneous_ev_specs', False):
        desired_soc_perc = env.config.get('ev', {}).get('desired_capacity', 1) * 100
        battery_cap = env.config.get('ev', {}).get('battery_capacity', 50)
        print("\n--- Desired SoC Summary ---")
        print("Homogeneous EV distribution:")
        print(f"  - All EVs have a battery capacity of {battery_cap} kWh.")
        print(f"  - The desired SoC for all EVs is {desired_soc_perc:.0f}%")
        print(f"  - This corresponds to a desired capacity of {battery_cap * desired_soc_perc / 100:.2f} kWh.")
        print("---------------------------\n")
        return

    summary_data = []
    desired_soc_perc = env.config.get('ev', {}).get('desired_capacity', 1)
    
    if hasattr(env, 'ev_specs'):
        for ev_name, specs in env.ev_specs.items():
            battery_cap = specs.get('battery_capacity', 0)
            desired_cap = battery_cap * desired_soc_perc
            summary_data.append({
                "EV Model": ev_name,
                "Battery Capacity (kWh)": battery_cap,
                "Desired SoC (%)": desired_soc_perc * 100,
                "Desired Capacity (kWh)": desired_cap
            })

    if not summary_data:
        print("Could not generate desired SoC summary.")
        return

    df_summary = pd.DataFrame(summary_data)
    
    print("\n--- Desired SoC Summary ---")
    print("Heterogeneous EV distribution:")
    print(df_summary.to_string(index=False))
    print("---------------------------\n")
