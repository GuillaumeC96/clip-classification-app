
FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requis
COPY backend_requirements.txt .
COPY azure_ml_backend.py .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r backend_requirements.txt

# Exposer le port
EXPOSE 5000

# Commande de démarrage
CMD ["python", "azure_ml_backend.py"]
