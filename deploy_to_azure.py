"""
Script pour déployer le backend CLIP corrigé sur Azure
"""

import os
import subprocess
import sys

def create_azure_deployment_files():
    """Créer les fichiers nécessaires pour le déploiement Azure"""
    
    # 1. Créer le fichier de configuration Azure
    azure_config = """
# Configuration Azure App Service
WEBSITES_PORT=5000
WEBSITES_ENABLE_APP_SERVICE_STORAGE=false
PYTHONPATH=/home/site/wwwroot
"""
    
    with open('.env', 'w') as f:
        f.write(azure_config)
    
    # 2. Créer le fichier de démarrage pour Azure
    startup_script = """#!/bin/bash
cd /home/site/wwwroot
python azure_ml_backend.py
"""
    
    with open('startup.sh', 'w') as f:
        f.write(startup_script)
    
    # 3. Créer le fichier de configuration Azure App Service
    web_config = """<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified"/>
    </handlers>
    <httpPlatform processPath="D:\home\Python39\python.exe"
                  arguments="D:\home\site\wwwroot\azure_ml_backend.py"
                  stdoutLogEnabled="true"
                  stdoutLogFile="D:\home\LogFiles\python.log"
                  startupTimeLimit="60"
                  startupRetryCount="3">
    </httpPlatform>
  </system.webServer>
</configuration>
"""
    
    with open('web.config', 'w') as f:
        f.write(web_config)
    
    print("✅ Fichiers de déploiement Azure créés")

def create_deployment_instructions():
    """Créer les instructions de déploiement"""
    
    instructions = """
# 🚀 Instructions de déploiement Azure App Service

## 1. Préparer le déploiement
```bash
# Installer Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Se connecter à Azure
az login

# Créer un groupe de ressources
az group create --name clip-backend-rg --location westeurope

# Créer un plan App Service
az appservice plan create --name clip-backend-plan --resource-group clip-backend-rg --sku B1 --is-linux

# Créer l'application web
az webapp create --resource-group clip-backend-rg --plan clip-backend-plan --name clip-backend-app --runtime "PYTHON|3.9"
```

## 2. Déployer le code
```bash
# Déployer via Git
az webapp deployment source config --name clip-backend-app --resource-group clip-backend-rg --repo-url https://github.com/GuillaumeC96/clip-classification-app.git --branch main --manual-integration

# Ou déployer via ZIP
zip -r backend.zip azure_ml_backend.py backend_requirements.txt web.config
az webapp deployment source config-zip --name clip-backend-app --resource-group clip-backend-rg --src backend.zip
```

## 3. Configurer les variables d'environnement
```bash
# Configurer le port
az webapp config appsettings set --name clip-backend-app --resource-group clip-backend-rg --settings WEBSITES_PORT=5000

# Configurer Python
az webapp config appsettings set --name clip-backend-app --resource-group clip-backend-rg --settings PYTHONPATH=/home/site/wwwroot
```

## 4. Tester le déploiement
```bash
# Obtenir l'URL
az webapp show --name clip-backend-app --resource-group clip-backend-rg --query defaultHostName --output tsv

# Tester la santé
curl https://clip-backend-app.azurewebsites.net/health
```

## 5. Mettre à jour le client
Une fois déployé, mettre à jour l'URL dans azure_client.py :
```python
'endpoint_url': "https://clip-backend-app.azurewebsites.net/score"
```
"""
    
    with open('DEPLOYMENT_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)
    
    print("✅ Instructions de déploiement créées dans DEPLOYMENT_INSTRUCTIONS.md")

def main():
    """Fonction principale"""
    print("🚀 Préparation du déploiement Azure")
    print("=" * 50)
    
    # Créer les fichiers de déploiement
    create_azure_deployment_files()
    create_deployment_instructions()
    
    print("\n✅ Fichiers de déploiement créés !")
    print("\n📋 Prochaines étapes :")
    print("1. Suivre les instructions dans DEPLOYMENT_INSTRUCTIONS.md")
    print("2. Déployer sur Azure App Service")
    print("3. Mettre à jour l'URL dans azure_client.py")
    print("4. Tester l'application cloud")
    
    return True

if __name__ == "__main__":
    main()
