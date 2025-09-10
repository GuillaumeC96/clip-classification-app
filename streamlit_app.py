"""
Application Streamlit Cloud - Classification de Produits
Point d'entrée principal pour le déploiement sur Streamlit Cloud
"""

import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Classification de Produits",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Redirection vers la page de prédiction
st.markdown("""
# 🔮 Application de Classification de Produits

Bienvenue dans l'application de classification de produits utilisant Azure ML PyTorch !

## 🚀 Fonctionnalités

- ✅ **Classification d'images** via Azure ML PyTorch
- ✅ **Prétraitement identique** au notebook de référence
- ✅ **Interface d'accessibilité** complète
- ✅ **Gestion robuste** des erreurs

## 📱 Navigation

Utilisez la sidebar pour naviguer entre les pages :
- **🔮 Prédiction** : Classification de produits
- **📊 EDA** : Analyse exploratoire des données
- **⚙️ Configuration** : Statut Azure ML

---

**🎯 Application prête pour la classification de produits !**
""")

# Lien vers la page de prédiction
if st.button("🔮 Aller à la page de prédiction", type="primary"):
    st.switch_page("pages/2_prediction.py")
