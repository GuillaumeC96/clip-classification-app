"""
Page d'accueil de l'application de classification de produits
Fichier principal pour Streamlit Cloud
"""

import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Classification de Produits",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page d'accueil
st.markdown("""
# 🔮 Application de Classification de Produits

Bienvenue dans l'application de classification de produits utilisant **Azure ML PyTorch** !

## 🚀 Fonctionnalités

- ✅ **Classification d'images** via Azure ML PyTorch
- ✅ **Prétraitement identique** au notebook de référence
- ✅ **Interface d'accessibilité** complète
- ✅ **Gestion robuste** des erreurs
- ✅ **1050 produits** dans la base de données

## 📱 Navigation

Utilisez la sidebar pour naviguer entre les pages :

### 🔮 **Prédiction**
- Classification de produits à partir d'images
- Prédiction de catégorie avec confiance
- Interface intuitive et accessible

### 📊 **EDA** 
- Analyse exploratoire des données
- Visualisation des catégories de produits
- Statistiques du dataset

### ⚙️ **Configuration**
- Statut de l'endpoint Azure ML
- Informations de débogage
- Configuration du système

## 🎯 **Comment utiliser l'application**

1. **Allez dans la page Prédiction** via la sidebar
2. **Uploadez une image** de produit ou utilisez le produit de test
3. **Remplissez les informations** (nom, marque, description, spécifications)
4. **Cliquez sur "Prédire la catégorie"**
5. **Consultez les résultats** avec la catégorie prédite et le niveau de confiance

---

**🎉 Application prête pour la classification de produits !**

*Utilisez la sidebar pour commencer votre analyse.*
""")

# Boutons de navigation
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔮 Page Prédiction", type="primary", use_container_width=True):
        st.switch_page("pages/2_prediction.py")

with col2:
    if st.button("📊 Analyse EDA", use_container_width=True):
        st.switch_page("pages/1_eda.py")

with col3:
    if st.button("⚙️ Configuration", use_container_width=True):
        st.switch_page("pages/3_configuration.py")

# Informations techniques
st.markdown("---")
st.info("""
**💡 Informations techniques :**
- **Backend** : Azure ML PyTorch (`http://localhost:5000/score`)
- **Modèle** : CLIP optimisé pour la classification de produits
- **Prétraitement** : Identique au notebook de référence
- **Dataset** : 1050 produits avec images et métadonnées
""")
