# 🆘 Guide de Dépannage Streamlit Cloud

## ✅ **Déploiement Réussi**
Si votre application se déploie correctement, vous verrez :
- ✅ "Your app is ready!"
- ✅ URL publique : `https://VOTRE-APP-NAME.streamlit.app`
- ✅ Interface accessible avec Azure ML ONNX

## 🚨 **Problèmes Courants et Solutions**

### **1. Erreur de Déploiement**
**Symptôme** : "Deployment failed"
**Solution** :
- Vérifier que le repository est public
- Vérifier que les fichiers principaux existent
- Vérifier que la branche `main` contient le code

### **2. Erreur 503 sur l'Application**
**Symptôme** : "Application Error" ou "Service Unavailable"
**Solution** :
- L'application utilise l'endpoint Azure ML de production
- Vérifier les logs de déploiement sur Streamlit Cloud
- Redémarrer l'application si nécessaire

### **3. Prédiction Non Disponible**
**Symptôme** : "Erreur lors de la prédiction"
**Solution** :
- L'endpoint Azure ML est configuré correctement
- Vérifier la connectivité réseau
- L'application utilise exclusivement Azure ML ONNX

### **4. Timeout de Déploiement**
**Symptôme** : Déploiement qui prend trop de temps
**Solution** :
- Le repository est optimisé
- Attendre 5-10 minutes maximum
- Redémarrer le déploiement si nécessaire

## 🔧 **Commandes de Vérification**

### **Vérifier le Repository Local**
```bash
cd /home/dev/Bureau/application_clean
git status
git log --oneline -3
```

### **Vérifier l'Endpoint Azure ML**
```bash
curl https://clip-onnx-interpretability.azurewebsites.net/health
```

### **Tester l'Application Localement**
```bash
streamlit run pages/2_prediction.py
```

## 📋 **Structure du Projet**
```
application_clean/
├── pages/
│   ├── 1_eda.py              # Analyse exploratoire
│   ├── 2_prediction.py       # Page de prédiction principale
│   └── 3_configuration.py    # Configuration Azure ML
├── azure_client.py           # Client Azure ML ONNX
├── accessibility.py          # Module d'accessibilité
├── produits_original.csv     # Dataset des produits
├── Images/                   # Images des produits
└── requirements.txt          # Dépendances Python
```

## 🎯 **Fonctionnalités Disponibles**
- ✅ Classification de produits via Azure ML ONNX
- ✅ Prétraitement identique au notebook
- ✅ Interface d'accessibilité
- ✅ Gestion des erreurs robuste
- ✅ Support multi-formats d'images

## 💡 **Conseils de Performance**
- L'application utilise l'endpoint Azure ML de production
- Les prédictions sont optimisées avec ONNX
- Le prétraitement est identique au notebook de référence
- Gestion automatique des erreurs de connectivité