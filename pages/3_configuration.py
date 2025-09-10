"""
Page de configuration Azure ML
Affiche le statut de l'endpoint Azure ML et les informations de débogage
"""

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_client import get_azure_client

# Importer le module d'accessibilité
from accessibility import init_accessibility_state, render_accessibility_sidebar, apply_accessibility_styles

# Initialiser l'état d'accessibilité
init_accessibility_state()

# Configuration de la page
st.set_page_config(
    page_title="Configuration Azure ML",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Configuration Azure ML")

# Afficher les options d'accessibilité dans la sidebar
render_accessibility_sidebar()

# Appliquer les styles d'accessibilité
apply_accessibility_styles()

st.markdown("---")

# Section 1: Statut actuel de la configuration
st.header("📊 Statut actuel de la configuration")

# Configuration de production
config = {
    'endpoint_url': "https://clip-onnx-interpretability.azurewebsites.net/score",
    'api_key': "dummy_key",
    'source': 'azure_ml_production'
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Configuration détectée")
    st.write(f"**Source:** {config['source']}")
    st.write(f"**Endpoint URL:** {config['endpoint_url']}")
    st.write(f"**API Key:** {'Configurée' if config['api_key'] else 'Non configurée'}")

with col2:
    st.subheader("🌐 Environnement")
    is_cloud = os.getenv('STREAMLIT_SERVER_ENVIRONMENT') == 'cloud'
    st.write(f"**Environnement:** {'Streamlit Cloud' if is_cloud else 'Développement'}")
    st.write(f"**Python:** {sys.version.split()[0]}")
    st.write(f"**Streamlit:** {st.__version__}")

# Section 2: Test du client Azure ML
st.header("🚀 Test du client Azure ML")

try:
    azure_client = get_azure_client(show_warning=True)
    
    st.success("✅ Client Azure ML initialisé avec succès")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Config Source:** {azure_client.config_source}")
        st.write(f"**Endpoint URL:** {azure_client.endpoint_url}")
        st.write(f"**Is ONNX:** {azure_client.is_onnx}")
    
    with col2:
        # Test de connectivité
        if st.button("🔗 Tester la connectivité"):
            with st.spinner("Test de connectivité en cours..."):
                status = azure_client.get_service_status()
                st.write(f"**Status:** {status['status']}")
                st.write(f"**Message:** {status['message']}")
                
                if status['status'] == 'healthy':
                    st.success("✅ Service Azure ML accessible")
                else:
                    st.warning("⚠️ Service non accessible")
    
except Exception as e:
    st.error(f"❌ Erreur lors de l'initialisation du client: {str(e)}")

# Section 3: Configuration pour Streamlit Cloud
st.header("☁️ Configuration pour Streamlit Cloud")

if is_cloud:
    st.info("🌐 Vous êtes sur Streamlit Cloud")
    st.markdown("""
    ### 📋 Configuration des secrets Streamlit Cloud
    
    Pour configurer l'endpoint Azure ML sur Streamlit Cloud :
    
    1. **Accédez à votre dashboard Streamlit Cloud**
    2. **Sélectionnez votre application**
    3. **Cliquez sur "Settings" puis "Secrets"**
    4. **Ajoutez les secrets suivants :**
    
    ```toml
    AZURE_ML_ENDPOINT_URL = "https://clip-onnx-interpretability.azurewebsites.net/score"
    AZURE_ML_API_KEY = "dummy_key"
    ```
    
    5. **Sauvegardez et redéployez l'application**
    """)
else:
    st.info("💻 Vous êtes en environnement de développement")
    st.markdown("""
    ### 🔧 Configuration locale
    
    Pour le développement local, l'application utilise la configuration par défaut.
    L'endpoint Azure ML de production est configuré automatiquement.
    """)

# Section 4: Informations sur l'endpoint
st.header("🔗 Informations sur l'endpoint")

st.markdown("""
### 📡 Endpoint Azure ML ONNX

**URL:** `https://clip-onnx-interpretability.azurewebsites.net/score`

**Type:** Modèle CLIP optimisé ONNX

**Fonctionnalités:**
- ✅ Classification d'images
- ✅ Analyse de texte
- ✅ Prédiction de catégories de produits
- ✅ Prétraitement identique au notebook

**Format de requête:**
```json
{
    "image": "base64_encoded_image",
    "text": "processed_text_keywords"
}
```

**Format de réponse:**
```json
{
    "predicted_category": "Watches",
    "confidence": 0.892,
    "source": "azure_onnx_simulation"
}
```
""")

# Section 5: Informations de débogage
st.header("🐛 Informations de débogage")

with st.expander("📋 Détails techniques"):
    st.code(f"""
    Environnement: {os.getenv('STREAMLIT_SERVER_ENVIRONMENT', 'développement')}
    Version Streamlit: {st.__version__}
    Python: {sys.version}
    Répertoire de travail: {os.getcwd()}
    
    Variables d'environnement:
    - AZURE_ML_ENDPOINT_URL: {os.getenv('AZURE_ML_ENDPOINT_URL', 'Non défini')}
    - AZURE_ML_API_KEY: {'Défini' if os.getenv('AZURE_ML_API_KEY') else 'Non défini'}
    """)

st.markdown("---")
st.info("💡 **Conseil :** L'application utilise l'endpoint Azure ML de production par défaut.")