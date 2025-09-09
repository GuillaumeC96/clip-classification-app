#!/bin/bash

# Script de configuration pour Streamlit Cloud
echo "🚀 Configuration de l'application CLIP..."

# Télécharger le modèle spaCy
echo "📦 Téléchargement du modèle spaCy..."
python -m spacy download en_core_web_trf

echo "✅ Configuration terminée !"
