# -------------------------------
# Dockerfile per Project_Master
# -------------------------------

# 1️⃣ Immagine base
FROM python:3.9-slim

# 2️⃣ Aggiorna pacchetti e installa cmake (necessario per sb3-contrib o simili)
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

# 3️⃣ Imposta la cartella di lavoro nel container
WORKDIR /app

# 4️⃣ Copia il file delle dipendenze
COPY requirements.txt .

# 5️⃣ Installa le dipendenze Python senza cache
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copia tutto il progetto nella cartella di lavoro
COPY . .

# 7️⃣ Esponi la porta su cui Streamlit girerà
EXPOSE 8501

# 8️⃣ Comando per avviare Streamlit su tutte le interfacce
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
c