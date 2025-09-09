# 🆘 Guide de Dépannage Streamlit Cloud

## ✅ **Déploiement Réussi**
Si votre application se déploie correctement, vous verrez :
- ✅ "Your app is ready!"
- ✅ URL publique : `https://VOTRE-APP-NAME.streamlit.app`
- ✅ Interface accessible avec interprétabilité ONNX

## 🚨 **Problèmes Courants et Solutions**

### **1. Erreur de Déploiement**
**Symptôme** : "Deployment failed"
**Solution** :
- Vérifier que le repository est public
- Vérifier que `accueil_streamlit_cloud.py` existe
- Vérifier que la branche `main` contient le code

### **2. Erreur 503 sur l'Application**
**Symptôme** : "Application Error" ou "Service Unavailable"
**Solution** :
- L'application utilise le bon endpoint Azure ML
- Vérifier les logs de déploiement sur Streamlit Cloud
- Redémarrer l'application si nécessaire

### **3. Interprétabilité Non Disponible**
**Symptôme** : "Interprétabilité non trouvée"
**Solution** :
- L'endpoint Azure ML est configuré correctement
- Les heatmaps et scores de mots-clés sont simulés
- Fonctionnalité disponible pour démonstration

### **4. Timeout de Déploiement**
**Symptôme** : Déploiement qui prend trop de temps
**Solution** :
- Le repository est optimisé (328 MB)
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

### **Tester Localement**
```bash
streamlit run accueil_streamlit_cloud.py
```

## 📞 **Support**
- **Streamlit Cloud** : [share.streamlit.io](https://share.streamlit.io)
- **GitHub Repository** : [github.com/GuillaumeC96/clip-classification-app](https://github.com/GuillaumeC96/clip-classification-app)
- **Logs de déploiement** : Disponibles sur Streamlit Cloud

## 🎯 **Résultat Attendu**
Application publique accessible avec :
- ✅ Classification de produits CLIP ONNX
- ✅ Interprétabilité (heatmaps + scores)
- ✅ Interface accessible et responsive
- ✅ Compatible plan gratuit Azure
