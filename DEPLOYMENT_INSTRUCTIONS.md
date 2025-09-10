
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
