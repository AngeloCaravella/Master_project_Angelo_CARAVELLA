import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# --- CONSTANTS FOR HEADERS ---
# Table 1: Infrastructure and Network
COL_FILE = "File"
COL_SCENARIO = "Scenario"
COL_SIM_DURATION = "Duration\n(hours)"
COL_STATIONS = "Stations"
COL_GRID_POWER = "Grid Power\n(kW)"
COL_NET_CAPACITY = "Net Capacity\n(kW)"
COL_MAX_DEMAND = "Max Demand\n(kW)"
COL_GRID_STRESS = "Grid\nStress"
COL_EV_TRAFFIC = "EV\nTraffic"
COL_DR = "Demand\nResponse"
COL_IL = "Inflexible\nLoads"
COL_PV = "Solar\nPower"

# Table 2: EV Details
COL_V2G = "V2G"
COL_V2G_PRICE_FACTOR = "V2G Price\nFactor"
COL_EFFICIENCY = "Efficiency\n(C/D)"
COL_HETEROGENEOUS_EV = "Heterogeneous\nEVs"
COL_DEFAULT_BATT = "Default Batt.\n(kWh)"
COL_EMERGENCY_BATT = "Emergency Cap.\n(kWh)"
COL_DESIRED_BATT = "Desired Cap.\n(%)"
COL_MIN_STAY = "Min. Stay\n(min)"


def get_nested_val(data: Dict[str, Any], keys: List[str], default: Any) -> Any:
    """Safely access a nested value in a dictionary."""
    temp_dict = data
    for key in keys:
        if isinstance(temp_dict, dict) and key in temp_dict:
            temp_dict = temp_dict[key]
        else:
            return default
    return temp_dict


def _process_single_config(file_path: str, file_name: str) -> Optional[Dict[str, Any]]:
    """Extracts and calculates ALL data from a single configuration file."""
    print(f"  -> Analyzing file: {file_name}")
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"WARNING: Could not process {file_name}. Error: {e}. Skipped.")
        return None

    # --- DATA FOR TABLE 1: INFRASTRUCTURE AND NETWORK ---
    num_stations = get_nested_val(config, ['number_of_charging_stations'], 0)
    transformer_power = get_nested_val(config, ['transformer', 'max_power'], 0)
    
    station_config = config.get('charging_station', {})
    max_current = station_config.get('max_charge_current', 0)
    voltage = station_config.get('voltage', 0)
    phases = station_config.get('phases', 0)
    max_station_power_kw = (voltage * max_current * (np.sqrt(3) if phases == 3 else 1)) / 1000

    total_potential_demand_kw = num_stations * max_station_power_kw
    capacity_ratio = total_potential_demand_kw / transformer_power if transformer_power > 0 else float('inf')

    timescale = get_nested_val(config, ['timescale'], 15)
    sim_length = get_nested_val(config, ['simulation_length'], 1)
    sim_duration_hours = (sim_length * timescale) / 60

    dr_included = get_nested_val(config, ['demand_response', 'include'], False)
    il_included = get_nested_val(config, ['inflexible_loads', 'include'], False)
    pv_included = get_nested_val(config, ['solar_power', 'include'], False)

    loads_power = transformer_power * get_nested_val(config, ['inflexible_loads', 'inflexible_loads_capacity_multiplier_mean'], 0) if il_included else 0
    solar_power = transformer_power * get_nested_val(config, ['solar_power', 'solar_power_capacity_multiplier_mean'], 0) if pv_included else 0
    net_capacity = transformer_power - loads_power + solar_power

    # --- DATA FOR TABLE 2: EV DETAILS ---
    ev_config = config.get('ev', {})
    charge_eff = get_nested_val(ev_config, ['charge_efficiency'], 1.0)
    discharge_eff = get_nested_val(ev_config, ['discharge_efficiency'], 1.0)
    
    return {
        COL_FILE: file_name.replace('.yaml', ''),
        COL_SCENARIO: str(config.get('scenario', 'N/A')).title(),
        COL_SIM_DURATION: f"{sim_duration_hours:.1f}",
        COL_STATIONS: num_stations,
        COL_GRID_POWER: transformer_power,
        COL_NET_CAPACITY: f"{net_capacity:.1f}",
        COL_MAX_DEMAND: f"{total_potential_demand_kw:.1f}",
        COL_GRID_STRESS: f"{capacity_ratio:.2f}",
        COL_EV_TRAFFIC: config.get('spawn_multiplier', 0),
        COL_DR: "YES" if dr_included else "NO",
        COL_IL: "YES" if il_included else "NO",
        COL_PV: "YES" if pv_included else "NO",
        
        COL_V2G: "YES" if config.get('v2g_enabled', False) else "NO",
        COL_V2G_PRICE_FACTOR: get_nested_val(config, ['discharge_price_factor'], 'N/A'),
        COL_EFFICIENCY: f"{charge_eff}/{discharge_eff}",
        COL_HETEROGENEOUS_EV: "YES" if get_nested_val(config, ['heterogeneous_ev_specs'], True) else "NO",
        COL_DEFAULT_BATT: get_nested_val(ev_config, ['battery_capacity'], 'N/A'),
        COL_EMERGENCY_BATT: get_nested_val(ev_config, ['min_emergency_battery_capacity'], 'N/A'),
        COL_DESIRED_BATT: f"{get_nested_val(ev_config, ['desired_capacity'], 1.0) * 100:.0f}",
        COL_MIN_STAY: get_nested_val(ev_config, ['min_time_of_stay'], 'N/A'),
    }

def _draw_grid_summary_table(ax, extracted_data: List[Dict[str, Any]]):
    """Draws the infrastructure and grid summary table on a given matplotlib axes."""
    cols_to_show = [COL_FILE, COL_SCENARIO, COL_SIM_DURATION, COL_STATIONS, COL_GRID_POWER, COL_NET_CAPACITY, COL_MAX_DEMAND, COL_GRID_STRESS, COL_EV_TRAFFIC, COL_DR, COL_IL, COL_PV]
    cell_data = [[str(row[col]) for col in cols_to_show] for row in extracted_data]
    
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_data, colLabels=cols_to_show, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            cell.set_facecolor('#ecf0f1' if i % 2 == 1 else 'white')
            if cols_to_show[j] == COL_NET_CAPACITY: cell.set_facecolor('#d9e2ec')
            if cols_to_show[j] == COL_GRID_STRESS:
                try:
                    ratio = float(cell.get_text().get_text())
                    if ratio > 1.05:
                        cell.set_facecolor('#e74c3c')
                        cell.set_text_props(weight='bold', color='white')
                    elif ratio > 0.95:
                        cell.set_facecolor('#f39c12')
                    else:
                        cell.set_facecolor('#2ecc71')
                except (ValueError, TypeError): pass
    ax.set_title('Infrastructure and Grid Summary', fontsize=18, pad=20)

def _draw_ev_details_table(ax, extracted_data: List[Dict[str, Any]]):
    """Draws the EV physical and economic details table on a given matplotlib axes."""
    cols_to_show = [COL_FILE, COL_V2G, COL_V2G_PRICE_FACTOR, COL_EFFICIENCY, COL_HETEROGENEOUS_EV, COL_DEFAULT_BATT, COL_EMERGENCY_BATT, COL_DESIRED_BATT, COL_MIN_STAY]
    cell_data = [[str(row[col]) for col in cols_to_show] for row in extracted_data]
    
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_data, colLabels=cols_to_show, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#34495e')
        else:
            cell.set_facecolor('#ecf0f1' if i % 2 == 1 else 'white')
            if cols_to_show[j] == COL_EFFICIENCY and cell.get_text().get_text() == "1.0/1.0":
                cell.set_facecolor('#f39c12')
            if cols_to_show[j] == COL_V2G_PRICE_FACTOR and cell.get_text().get_text() == "1":
                cell.set_facecolor('#f39c12')
    ax.set_title('EV Physical and Economic Details', fontsize=18, pad=20)

def plot_combined_summary(extracted_data: List[Dict[str, Any]], save_fig: bool = False) -> plt.Figure:
    """Generates a single figure containing both summary tables."""
    num_rows = len(extracted_data)
    fig_height = (num_rows * 0.5 + 1.5) * 2 + 2
    
    fig, axes = plt.subplots(2, 1, figsize=(28, fig_height))

    _draw_grid_summary_table(axes[0], extracted_data)
    _draw_ev_details_table(axes[1], extracted_data)

    fig.suptitle('Configuration Parameters Summary', fontsize=24, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_fig:
        plt.savefig("Configuration_Summary.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
        print("\nCombined summary table 'Configuration_Summary.png' saved successfully.")
    
    return fig

def generate_summary_figures_for_streamlit(config_directory: str, file_list: List[str]) -> List[plt.Figure]:
    """Streamlit function: orchestrates the analysis and returns the figures."""
    file_names_only = [os.path.basename(p) for p in file_list]
    
    extracted_data = [
        data for i, file_path in enumerate(file_list)
        if (data := _process_single_config(file_path, file_names_only[i])) is not None
    ]
    
    if not extracted_data:
        return []

    fig = plot_combined_summary(extracted_data, save_fig=False)
    return [fig]

def analyze_configs_and_save(config_directory: str, file_list: List[str]):
    """Main function that orchestrates the analysis and image creation."""
    print("Starting detailed analysis of configuration files...")
    extracted_data = [
        data for file_name in file_list
        if (data := _process_single_config(os.path.join(config_directory, file_name), file_name)) is not None
    ]
    
    if not extracted_data:
        print("\nNo valid data extracted. Cannot generate images.")
        return

    fig = plot_combined_summary(extracted_data, save_fig=True)
    plt.close(fig)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    CONFIG_DIR = os.path.join("ev2gym", "example_config_files")
    
    if not os.path.isdir(CONFIG_DIR):
        print(f"ERROR: Directory '{CONFIG_DIR}' not found.")
    else:
        files_to_analyze = sorted([f for f in os.listdir(CONFIG_DIR) if f.endswith('.yaml')])
        if not files_to_analyze:
            print(f"No .yaml files found in '{CONFIG_DIR}'.")
        else:
            print(f"Found {len(files_to_analyze)} config files to analyze in '{CONFIG_DIR}':")
            for f in files_to_analyze:
                print(f"  - {f}")
            analyze_configs_and_save(CONFIG_DIR, files_to_analyze)