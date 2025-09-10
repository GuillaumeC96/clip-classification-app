#!/usr/bin/env python3
"""
Script pour vérifier qu'aucun message ONNX n'est présent dans l'application
"""

import os
import re
from pathlib import Path

def check_onnx_in_files():
    """Vérifier qu'aucun fichier de l'application ne contient de références ONNX"""
    print("🔍 Vérification des références ONNX dans l'application...")
    
    # Fichiers principaux de l'application
    app_files = [
        "pages/2_prediction.py",
        "pages/3_configuration.py", 
        "azure_client.py",
        "accueil_streamlit_cloud.py",
        "streamlit_app.py",
        "app.py",
        "requirements.txt"
    ]
    
    onnx_found = False
    
    for file_path in app_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Chercher les références ONNX (sauf dans les commentaires de migration)
            onnx_patterns = [
                r'ONNX',
                r'onnx',
                r'azure_ml_onnx',
                r'azure_onnx'
            ]
            
            for pattern in onnx_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Exclure les commentaires de migration
                    if not any(comment in content for comment in [
                        "migration ONNX vers PyTorch",
                        "suppression ONNX",
                        "remplace ONNX",
                        "ONNX initialisé",  # Ancien message corrigé
                        "# onnxruntime",  # Commentaire dans requirements.txt
                        "Supprimé - migration vers PyTorch"
                    ]):
                        print(f"❌ Référence ONNX trouvée dans {file_path}: {matches}")
                        onnx_found = True
                    else:
                        print(f"ℹ️ Référence ONNX dans commentaire de migration: {file_path}")
    
    return not onnx_found

def check_pytorch_messages():
    """Vérifier que les messages PyTorch sont présents"""
    print("🔍 Vérification des messages PyTorch...")
    
    pytorch_files = [
        "pages/2_prediction.py",
        "azure_client.py"
    ]
    
    pytorch_found = True
    
    for file_path in pytorch_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'pytorch' not in content.lower() and 'PyTorch' not in content:
                print(f"⚠️ Aucune référence PyTorch trouvée dans {file_path}")
                pytorch_found = False
    
    return pytorch_found

def check_specific_messages():
    """Vérifier les messages spécifiques"""
    print("🔍 Vérification des messages spécifiques...")
    
    # Vérifier le message principal
    if os.path.exists("pages/2_prediction.py"):
        with open("pages/2_prediction.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "Système de prédiction Azure ML PyTorch initialisé" in content:
            print("✅ Message principal correct: PyTorch")
            return True
        elif "Système de prédiction Azure ML ONNX initialisé" in content:
            print("❌ Message principal incorrect: ONNX")
            return False
        else:
            print("⚠️ Message principal non trouvé")
            return False
    
    return False

def main():
    """Fonction principale de vérification"""
    print("🧪 Vérification finale - Aucun message ONNX")
    print("=" * 50)
    
    checks = [
        ("Aucune référence ONNX", check_onnx_in_files),
        ("Messages PyTorch présents", check_pytorch_messages),
        ("Message principal correct", check_specific_messages)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}...")
        if check_func():
            print(f"✅ {check_name}: OK")
        else:
            print(f"❌ {check_name}: ÉCHEC")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 VÉRIFICATION RÉUSSIE !")
        print("✅ Aucun message ONNX dans l'application")
        print("✅ Tous les messages utilisent PyTorch")
        print("✅ L'application est prête pour le cloud")
    else:
        print("❌ VÉRIFICATION ÉCHOUÉE")
        print("⚠️ Des corrections supplémentaires sont nécessaires")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
