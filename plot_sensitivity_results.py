
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- UI Helper Functions ---

def get_user_input(prompt, default=None):
    user_input = input(f"{prompt} (default: {default}): ")
    return user_input or default

def select_from_list(items, prompt, multiple=False, default_choice=1):
    print(f"\n{prompt}")
    for i, item in enumerate(items):
        print(f"  {i+1}. {os.path.basename(item)}")
    
    if not items:
        print("No items available.")
        return None if not multiple else []

    if multiple:
        choices_str = get_user_input(f"Select one or more (e.g., '1, 3', 'all')", 'all').lower()
        if 'all' in choices_str:
            return items
        try:
            cleaned_str = choices_str.replace(',', ' ')
            indices = [int(i.strip()) - 1 for i in cleaned_str.split() if i.strip().isdigit()]
            if not indices:
                raise ValueError("No valid indices found.")
            return [items[i] for i in indices if 0 <= i < len(items)]
        except (ValueError, IndexError):
            print("Invalid selection. Returning all items.")
            return items
    else:
        try:
            choice_str = get_user_input(f"Choice", str(default_choice))
            choice = int(choice_str) - 1
            return items[choice] if 0 <= choice < len(items) else items[default_choice - 1]
        except (ValueError, IndexError):
            return items[default_choice - 1]

def navigate_and_select_files(start_path, file_pattern):
    """An interactive file navigator to select one or more files."""
    current_path = os.path.abspath(start_path)
    selected_files = []

    while True:
        print(f"\nCurrent Directory: {current_path}")
        try:
            content = sorted(os.listdir(current_path))
            dirs = [d for d in content if os.path.isdir(os.path.join(current_path, d))]
            files = [f for f in content if os.path.isfile(os.path.join(current_path, f)) and f.endswith(file_pattern)]
        except FileNotFoundError:
            print("Directory not found. Returning to parent.")
            current_path = os.path.dirname(current_path)
            continue

        print("\nSubdirectories:")
        for i, dirname in enumerate(dirs):
            print(f"  {i+1}. {dirname}/")
        
        print(f"\nFiles matching '{file_pattern}':")
        for i, filename in enumerate(files):
            print(f"  {len(dirs)+i+1}. {filename}")

        print("\nOptions:")
        print("  .. (Go up)")
        print("  s (Finish selection)")

        choice = input("\nEnter a number to select/deselect, 's' to finish, or '..' to go up: ").lower()

        if choice == '..':
            current_path = os.path.dirname(current_path)
        elif choice == 's':
            if not selected_files:
                print("No files selected.")
                return []
            return selected_files
        else:
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(dirs):
                    current_path = os.path.join(current_path, dirs[choice_idx])
                elif len(dirs) <= choice_idx < len(dirs) + len(files):
                    file_path = os.path.join(current_path, files[choice_idx - len(dirs)])
                    if file_path in selected_files:
                        print(f"Deselected: {os.path.basename(file_path)}")
                        selected_files.remove(file_path)
                    else:
                        print(f"Selected: {os.path.basename(file_path)}")
                        selected_files.append(file_path)
                else:
                    print("Invalid number.")
            except ValueError:
                print("Invalid input.")

# --- Plotting Functions ---

def plot_morris_flow():
    """Handles the workflow for plotting Morris analysis results."""
    print("\n--- Plot Morris Analysis Results ---")
    start_path = './sensitivity_analysis_results/'
    if not os.path.exists(start_path):
        print(f"ERROR: Directory '{start_path}' not found.")
        return

    selected_files = navigate_and_select_files(start_path, file_pattern=".csv")
    if not selected_files:
        return

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", len(selected_files))

    for i, file_path in enumerate(selected_files):
        try:
            df = pd.read_csv(file_path)
            algo_name = os.path.basename(file_path).replace('morris_indices_', '').replace('.csv', '')
            
            if 'mu_star' not in df.columns or 'sigma' not in df.columns or 'names' not in df.columns:
                print(f"Skipping {file_path}: missing required columns (mu_star, sigma, names).")
                continue

            plt.scatter(df['mu_star'], df['sigma'], label=algo_name, color=palette[i], s=100, alpha=0.8)
            for _, row in df.iterrows():
                plt.text(row['mu_star'] + 0.01, row['sigma'], row['names'], fontsize=9)
        except Exception as e:
            print(f"Could not process file {file_path}. Error: {e}")

    plt.title('Morris Sensitivity Analysis Comparison')
    plt.xlabel('μ* (Total Influence)')
    plt.ylabel('σ (Interactions & Non-linearities)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(start_path, 'morris_comparison_plot.png')
    plt.savefig(save_path)
    print(f"\nComparison plot saved to: {save_path}")
    plt.show()

def plot_summary_flow():
    """Handles the workflow for plotting OAT and Scenario Comparison results."""
    print("\n--- Plot OAT / Scenario Comparison Results ---")
    start_path = './sensitivity_analysis_results/'
    if not os.path.exists(start_path):
        print(f"ERROR: Directory '{start_path}' not found.")
        return

    selected_files = navigate_and_select_files(start_path, file_pattern="summary.csv")
    if not selected_files:
        return

    file_path = selected_files[0]
    if len(selected_files) > 1:
        print("Warning: Multiple files selected. Only the first one will be used.")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Could not read file {file_path}. Error: {e}")
        return

    all_algorithms = sorted(df['Algorithm'].unique())
    selected_algorithms = select_from_list(all_algorithms, "Select algorithms to plot:", multiple=True)
    if not selected_algorithms:
        return

    df = df[df['Algorithm'].isin(selected_algorithms)]

    # Detect analysis type
    if 'parameter_value' in df.columns:
        # OAT Analysis
        param_name = df['parameter_name'].iloc[0]
        metrics = [col.replace('_mean', '') for col in df.columns if col.endswith('_mean')]
        selected_metric = select_from_list(metrics, "Select metric to plot:")

        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='parameter_value', y=f'{selected_metric}_mean', hue='Algorithm', marker='o', style='Algorithm')
        plt.title(f"Sensitivity of '{selected_metric}' to '{param_name}'")
        plt.xlabel(param_name)
        plt.ylabel(selected_metric)
        plt.grid(True)
        plt.legend(title='Algorithm')
        save_path = os.path.join(os.path.dirname(file_path), f'OAT_plot_{selected_metric}.png')
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
        plt.show()

    elif 'scenario_name' in df.columns:
        # Scenario Comparison Analysis
        metrics = [col.replace('_mean', '') for col in df.columns if col.endswith('_mean')]
        selected_metric = select_from_list(metrics, "Select metric to plot:")

        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x='scenario_name', y=f'{selected_metric}_mean', hue='Algorithm')
        plt.title(f"Comparison of '{selected_metric}' across Scenarios")
        plt.xlabel('Scenario')
        plt.ylabel(selected_metric)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Algorithm')
        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(file_path), f'ScenarioComp_plot_{selected_metric}.png')
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
        plt.show()
    else:
        print("Could not determine analysis type from CSV columns.")

# --- Main Function ---

def main():
    """Main function to drive the plotting script."""
    while True:
        print("\n--- Sensitivity Analysis Plotter ---")
        choice = get_user_input("Select the type of result to plot:\n  1. Morris Analysis\n  2. OAT / Scenario Comparison", "1")
        
        if choice == '1':
            plot_morris_flow()
        elif choice == '2':
            plot_summary_flow()
        else:
            print("Invalid choice.")

        if get_user_input("\nPlot another result? (y/n)", "n").lower() != 'y':
            break

if __name__ == "__main__":
    main()
