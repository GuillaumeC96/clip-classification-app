"""
Page de configuration Azure ML pour Streamlit Cloud
"""

import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_client import get_azure_client
# from streamlit_secrets_config import get_azure_config  # Module supprimé

# Importer le module d'accessibilité
from accessibility import init_accessibility_state, render_accessibility_sidebar, apply_accessibility_styles

# Initialiser l'état d'accessibilité
init_accessibility_state()

st.title("⚙️ Configuration Azure ML")

# Afficher les options d'accessibilité dans la sidebar
render_accessibility_sidebar()

# Appliquer les styles d'accessibilité
apply_accessibility_styles()

st.markdown("---")

# Section 1: Statut actuel de la configuration
st.header("📊 Statut actuel de la configuration")

# Configuration par défaut (streamlit_secrets_config supprimé)
config = {
    'endpoint_url': "https://your-endpoint.westeurope.inference.ml.azure.com/score",
    'api_key': "your_api_key_here",
    'source': 'default_hardcoded'
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Configuration détectée")
    st.write(f"**Source:** {config['source']}")
    st.write(f"**Endpoint URL:** {config['endpoint_url'] or 'Non configuré'}")
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
        st.write(f"**Use Simulated:** {azure_client.use_simulated}")
        st.write(f"**Is Simulated:** {azure_client.is_simulated}")
        
        # Test de connectivité
        if st.button("🔗 Tester la connectivité"):
            with st.spinner("Test de connectivité en cours..."):
                status = azure_client.get_service_status()
                st.write(f"**Status:** {status['status']}")
                st.write(f"**Message:** {status['message']}")
                
                if status['status'] == 'healthy':
                    st.success("✅ Service Azure ML accessible")
                elif status['status'] == 'simulated':
                    st.info("ℹ️ Mode simulé activé")
                else:
                    st.warning("⚠️ Service non accessible")
    
except Exception as e:
    st.error(f"❌ Erreur lors de l'initialisation du client: {str(e)}")

# Section 3: Configuration pour Streamlit Cloud
st.header("☁️ Configuration pour Streamlit Cloud")

if is_cloud:
    st.info("🌐 Vous êtes sur Streamlit Cloud")
    
    if config['source'] == 'streamlit_secrets':
        st.success("✅ Configuration Azure ML détectée dans les secrets Streamlit Cloud")
        st.info("Votre application est correctement configurée !")
    else:
        st.warning("⚠️ Configuration Azure ML non détectée dans les secrets Streamlit Cloud")
        
        st.subheader("📋 Étapes de configuration :")
        
        st.markdown("""
        1. **Allez sur [Streamlit Cloud](https://share.streamlit.io/)**
        2. **Sélectionnez votre application**
        3. **Cliquez sur 'Settings' (⚙️)**
        4. **Cliquez sur 'Secrets'**
        5. **Ajoutez ces secrets :**
        """)
        
        st.code("""
AZURE_ML_ENDPOINT_URL = "https://your-endpoint.westeurope.inference.ml.azure.com/score"
AZURE_ML_API_KEY = "your_api_key_here"
        """)
        
        st.markdown("""
        6. **Cliquez sur 'Save'**
        7. **Attendez le redéploiement automatique**
        """)
        
        st.info("💡 Remplacez `your-endpoint` et `your_api_key_here` par vos vraies valeurs Azure ML")
        
else:
    st.info("💻 Vous êtes en développement")
    
    if config['source'] == 'env_vars':
        st.success("✅ Configuration Azure ML détectée dans les variables d'environnement")
    else:
        st.info("ℹ️ Configuration par défaut utilisée")
        
        st.subheader("📋 Configuration développement :")
        st.markdown("""
        Pour configurer Azure ML en développement, créez un fichier `.env_azure_production` :
        """)
        
        st.code("""
AZURE_ML_ENDPOINT_URL=https://your-endpoint.westeurope.inference.ml.azure.com/score
AZURE_ML_API_KEY=your_api_key_here
        """)

# Section 4: Informations sur les endpoints Azure ML
st.header("🔗 Types d'endpoints Azure ML")

st.markdown("""
### 🎯 Endpoints Azure ML supportés :

1. **Azure ML Managed Endpoint** (recommandé)
   ```
   https://your-endpoint.westeurope.inference.ml.azure.com/score
   ```

2. **Azure Container Instance (ACI)**
   ```
   https://your-endpoint.westeurope.azurecontainer.io/score
   ```

3. **Azure Kubernetes Service (AKS)**
   ```
   https://your-endpoint.westeurope.cloudapp.azure.com/score
   ```

4. **Azure App Service**
   ```
   https://your-endpoint.azurewebsites.net/api/predict
   ```
""")

# Section 5: Dépannage
st.header("🔧 Dépannage")

with st.expander("❓ Problèmes courants et solutions"):
    st.markdown("""
    ### 🚨 "AZURE_ML_ENDPOINT_URL non configuré"
    
    **Cause :** Les secrets Azure ML ne sont pas configurés dans Streamlit Cloud
    
    **Solution :**
    1. Vérifiez que les secrets sont correctement configurés
    2. Redéployez l'application
    3. Vérifiez les logs de déploiement
    
    ### 🌐 "Service non accessible"
    
    **Cause :** L'endpoint Azure ML n'est pas accessible
    
    **Solutions :**
    1. Vérifiez que l'endpoint est déployé et actif
    2. Vérifiez la clé API
    3. Testez l'endpoint directement avec curl ou Postman
    
    ### 🔑 "Erreur d'authentification"
    
    **Cause :** Clé API incorrecte ou expirée
    
    **Solution :**
    1. Régénérez la clé API dans Azure ML
    2. Mettez à jour les secrets Streamlit Cloud
    
    ### ⚡ "Timeout"
    
    **Cause :** L'endpoint met trop de temps à répondre
    
    **Solutions :**
    1. Vérifiez les performances de l'endpoint
    2. Augmentez le timeout dans la configuration
    3. Optimisez le modèle ONNX
    """)

# Section 6: Informations de débogage
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
    - USE_SIMULATED_MODEL: {os.getenv('USE_SIMULATED_MODEL', 'Non défini')}
    """)

st.markdown("---")
st.info("💡 **Conseil :** Après avoir configuré les secrets, redéployez votre application pour que les changements prennent effet.")
