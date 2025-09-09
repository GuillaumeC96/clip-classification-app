# 🎯 Classification de Produits avec CLIP - Application Web

## 📋 Description

Application de classification de produits e-commerce utilisant un modèle CLIP fine-tuné. L'application permet de classifier automatiquement des produits en analysant leurs images et descriptions textuelles.

## 🚀 Déploiement sur Streamlit Cloud

### Prérequis
- Compte GitHub
- Compte Streamlit Cloud (gratuit)

### Étapes de déploiement

1. **Forker ce repository** sur GitHub
2. **Aller sur [Streamlit Cloud](https://share.streamlit.io/)**
3. **Connecter votre repository GitHub**
4. **Configurer le déploiement** :
   - **Repository** : Votre fork
   - **Branch** : main
   - **Main file path** : `accueil_cloud.py`
5. **Déployer**

### Configuration automatique

L'application est configurée pour fonctionner en mode cloud sur Streamlit Cloud :
- ✅ Modèle CLIP pré-entraîné
- ✅ Traitement de texte avec spaCy
- ✅ Interface accessible
- ✅ Options d'accessibilité

## 🌐 Accès à l'application

Une fois déployée, l'application sera accessible via une URL publique du type :
`https://your-app-name.streamlit.app`

## 🎯 Fonctionnalités

- **📊 Analyse Exploratoire des Données (EDA)** : Visualisations interactives
- **🔮 Prédiction de Catégorie** : Classification de produits
- **👁️ Accessibilité** : Options de contraste et mode daltonien
- **📱 Interface responsive** : Compatible mobile et desktop

## 🔧 Configuration technique

- **Frontend** : Streamlit
- **Backend** : Modèle CLIP ONNX sur Azure ML
- **Traitement de texte** : spaCy
- **Visualisations** : Plotly, Matplotlib

## 📊 Catégories supportées

- 👶 Baby Care
- 💄 Beauty and Personal Care
- 💻 Computers
- 🏠 Home Decor & Festive Needs
- 🛋️ Home Furnishing
- 🍽️ Kitchen & Dining
- ⌚ Watches

## 🆘 Support

Pour toute question ou problème :
1. Vérifier les logs de déploiement sur Streamlit Cloud
2. Consulter la documentation Streamlit
3. Créer une issue sur GitHub

---

**🎉 Application prête pour le déploiement public !**