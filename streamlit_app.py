import streamlit as st
import os
import sys
import subprocess
from glob import glob

# Aggiungi la directory del progetto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

st.set_page_config(layout="wide", page_title="EV2Gym Dashboard")

st.title("EV2Gym Dashboard")
st.markdown("Visualizza i risultati delle simulazioni, analizza le configurazioni e calibra il modello di batteria.")

st.sidebar.markdown("---")
st.sidebar.markdown("Sviluppato da: **Angelo Caravella**")

# --- Esegui Fit_battery.py ---
if st.sidebar.button("Calibra Modello Batteria (Fit_battery.py)"):
    st.subheader("Esecuzione di Fit_battery.py")
    with st.spinner("Calibrazione in corso..."):
        try:
            process = subprocess.run(["python", os.path.join(project_root, "Fit_battery.py")], capture_output=True, text=True, check=True)
            st.success("Calibrazione completata con successo!")
            st.code(process.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"Errore durante l'esecuzione di Fit_battery.py (codice {e.returncode}):")
            st.code(e.stderr)
        except FileNotFoundError:
            st.error("Errore: Lo script 'Fit_battery.py' non è stato trovato.")

# --- NUOVA SEZIONE: ANALISI CONFIGURAZIONI --
config_path = os.path.join(project_root, "ev2gym", "example_config_files")
available_scenarios_full_paths = sorted(glob(os.path.join(config_path, "*.yaml")))
available_scenario_names = [os.path.basename(s).replace(".yaml", "") for s in available_scenarios_full_paths]

with st.expander("Analisi Configurazioni Scenario (.yaml)"):
    st.markdown("Visualizza tabelle riassuntive dei parametri per gli scenari selezionati.")
    
    selected_scenario_names = st.multiselect(
        "Seleziona gli scenari da analizzare:",
        options=available_scenario_names,
        default=[]
    )
    scenarios_to_analyze = [s for s in available_scenarios_full_paths if os.path.basename(s).replace(".yaml", "") in selected_scenario_names]

    if st.button("Genera e Visualizza Tabelle Riassuntive"):
        if not scenarios_to_analyze:
            st.warning("Per favore, seleziona almeno uno scenario dalla lista.")
        else:
            with st.spinner("Analisi dei file di configurazione in corso..."):
                try:
                    from Compare import generate_summary_figures_for_streamlit
                    
                    summary_figures = generate_summary_figures_for_streamlit(config_path, scenarios_to_analyze)
                    
                    if not summary_figures:
                        st.error("Impossibile generare le tabelle. Nessun dato valido estratto dai file.")
                    else:
                        st.success(f"Tabelle generate con successo per {len(scenarios_to_analyze)} scenari.")
                        for fig in summary_figures:
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"Si è verificato un errore durante la generazione delle tabelle: {e}")


# =============================================================================
# --- Visualizzatore Risultati ---
# =============================================================================
st.subheader("Visualizzatore Risultati")

results_base_path = os.path.join(project_root, "results")

if not os.path.exists(results_base_path):
    st.warning(f"La cartella risultati '{results_base_path}' non esiste ancora.")
else:
    benchmark_folders = [f.name for f in os.scandir(results_base_path) if f.is_dir()]
    benchmark_folders.sort(reverse=True)

    if not benchmark_folders:
        st.info("Nessuna cartella di benchmark trovata nella directory risultati.")
    else:
        selected_benchmark_folder = st.selectbox(
            "Seleziona una cartella di benchmark:",
            options=benchmark_folders
        )

        if selected_benchmark_folder:
            selected_folder_path = os.path.join(results_base_path, selected_benchmark_folder)
            
            sub_folders = sorted([f.name for f in os.scandir(selected_folder_path) if f.is_dir()])
            
            sub_folder_options = ["Tutte le sottocartelle"] + sub_folders
            
            selected_sub_folder = st.selectbox(
                "Seleziona una sottocartella (o tutte):",
                options=sub_folder_options
            )

            if selected_sub_folder == "Tutte le sottocartelle":
                search_path = selected_folder_path
                display_caption_base = selected_folder_path
            else:
                search_path = os.path.join(selected_folder_path, selected_sub_folder)
                display_caption_base = search_path

            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
            all_image_paths = []
            for root, _, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        all_image_paths.append(os.path.join(root, file))
            all_image_paths.sort()

            if not all_image_paths:
                st.info(f"Nessuna immagine trovata in '{selected_sub_folder}'.")
            else:
                for img_path in all_image_paths:
                    relative_path = os.path.relpath(img_path, display_caption_base)
                    st.image(img_path, caption=relative_path, use_container_width=True)