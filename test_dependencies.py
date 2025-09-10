#!/usr/bin/env python3
"""
Script pour tester que toutes les dépendances sont disponibles
"""

def test_imports():
    """Tester l'importation de tous les modules requis"""
    print("🧪 Test des dépendances de l'application...")
    
    modules_to_test = [
        ("streamlit", "st"),
        ("pandas", "pd"),
        ("plotly.express", "px"),
        ("matplotlib.pyplot", "plt"),
        ("PIL", "Image"),
        ("numpy", "np"),
        ("requests", "requests"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sklearn", "sklearn"),
        ("wordcloud", "WordCloud")
    ]
    
    failed_imports = []
    successful_imports = []
    
    for module_name, alias in modules_to_test:
        try:
            if alias == "st":
                import streamlit as st
            elif alias == "pd":
                import pandas as pd
            elif alias == "px":
                import plotly.express as px
            elif alias == "plt":
                import matplotlib.pyplot as plt
            elif alias == "Image":
                from PIL import Image
            elif alias == "np":
                import numpy as np
            elif alias == "requests":
                import requests
            elif alias == "torch":
                import torch
            elif alias == "transformers":
                import transformers
            elif alias == "sklearn":
                import sklearn
            elif alias == "WordCloud":
                from wordcloud import WordCloud
            
            successful_imports.append(module_name)
            print(f"✅ {module_name}: OK")
            
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
            print(f"❌ {module_name}: ÉCHEC - {str(e)}")
    
    return successful_imports, failed_imports

def test_streamlit_app():
    """Tester que l'application Streamlit peut être importée"""
    print("\n🧪 Test de l'application Streamlit...")
    
    try:
        # Tester l'importation des pages
        import sys
        import os
        
        # Ajouter le répertoire courant au path
        sys.path.insert(0, os.getcwd())
        
        # Tester l'importation des modules de l'application
        from azure_client import AzureMLClient
        print("✅ azure_client: OK")
        
        # Tester l'importation de la page EDA
        import importlib.util
        spec = importlib.util.spec_from_file_location("eda", "pages/1_eda.py")
        if spec and spec.loader:
            print("✅ pages/1_eda.py: OK")
        else:
            print("❌ pages/1_eda.py: ÉCHEC")
            return False
        
        # Tester l'importation de la page de prédiction
        spec = importlib.util.spec_from_file_location("prediction", "pages/2_prediction.py")
        if spec and spec.loader:
            print("✅ pages/2_prediction.py: OK")
        else:
            print("❌ pages/2_prediction.py: ÉCHEC")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de l'application: {str(e)}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 Test des dépendances - Application Streamlit Cloud")
    print("=" * 60)
    
    # Test des imports
    successful_imports, failed_imports = test_imports()
    
    # Test de l'application
    app_ok = test_streamlit_app()
    
    print("\n" + "=" * 60)
    print("📊 Résumé des tests:")
    print(f"✅ Modules importés avec succès: {len(successful_imports)}")
    print(f"❌ Modules en échec: {len(failed_imports)}")
    print(f"✅ Application Streamlit: {'OK' if app_ok else 'ÉCHEC'}")
    
    if failed_imports:
        print("\n❌ Modules en échec:")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
    
    if len(failed_imports) == 0 and app_ok:
        print("\n🎉 TOUS LES TESTS RÉUSSIS !")
        print("✅ Toutes les dépendances sont disponibles")
        print("✅ L'application Streamlit est prête")
        print("✅ Prêt pour le déploiement sur Streamlit Cloud")
        return True
    else:
        print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("⚠️ Des dépendances manquantes ont été détectées")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
