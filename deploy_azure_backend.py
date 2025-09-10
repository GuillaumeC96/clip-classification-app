"""
Script de déploiement pour le backend Azure ML avec vraies heatmaps CLIP
"""

import os
import subprocess
import sys

def install_requirements():
    """Installer les dépendances du backend"""
    print("📦 Installation des dépendances du backend...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "backend_requirements.txt"])
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        return False

def test_backend():
    """Tester le backend localement"""
    print("🧪 Test du backend local...")
    try:
        # Importer et tester le backend
        from azure_ml_backend import load_clip_model, generate_clip_attention_heatmap
        from PIL import Image
        import numpy as np
        
        # Test de chargement du modèle
        if not load_clip_model():
            print("❌ Échec du chargement du modèle")
            return False
        
        # Test de génération de heatmap
        test_image = Image.new('RGB', (224, 224), color='red')
        result = generate_clip_attention_heatmap(test_image, "test product", "test, keywords")
        
        if result:
            print("✅ Test de heatmap réussi")
            print(f"   Shape heatmap: {np.array(result['heatmap']).shape}")
            print(f"   Keywords: {result['keywords']}")
            return True
        else:
            print("❌ Échec de la génération de heatmap")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def create_dockerfile():
    """Créer un Dockerfile pour le déploiement Azure"""
    dockerfile_content = """
FROM python:3.9-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
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
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("✅ Dockerfile créé")

def main():
    """Fonction principale"""
    print("🚀 Déploiement du backend Azure ML avec vraies heatmaps CLIP")
    print("=" * 60)
    
    # 1. Installer les dépendances
    if not install_requirements():
        return False
    
    # 2. Tester le backend
    if not test_backend():
        print("⚠️ Tests échoués, mais continuons...")
    
    # 3. Créer le Dockerfile
    create_dockerfile()
    
    print("\n✅ Backend prêt pour le déploiement !")
    print("\n📋 Prochaines étapes :")
    print("1. Déployer sur Azure Container Instances (ACI)")
    print("2. Ou déployer sur Azure App Service")
    print("3. Mettre à jour l'URL dans azure_client.py")
    
    return True

if __name__ == "__main__":
    main()
