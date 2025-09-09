"""
Page de prédiction pour la version cloud avec Azure ML
"""

import os
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from azure_client import get_azure_client

# Importer le module d'accessibilité
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from accessibility import init_accessibility_state, render_accessibility_sidebar, apply_accessibility_styles

# Initialiser l'état d'accessibilité
init_accessibility_state()

st.title("🔮 Prédiction de Catégorie")

# Client Azure ML
azure_client = get_azure_client()

# Afficher les options d'accessibilité dans la sidebar
render_accessibility_sidebar()

# Appliquer les styles d'accessibilité
apply_accessibility_styles()

st.markdown("---")

# Fonction pour charger le produit de test par défaut
@st.cache_data
def load_default_test_product():
    """Charger le produit de test par défaut"""
    try:
        # Charger les données des produits
        df = pd.read_csv('produits_original.csv')
        
        # Produit de test par défaut (montre)
        test_product_id = '1120bc768623572513df956172ffefeb'
        product = df[df['uniq_id'] == test_product_id]
        
        if not product.empty:
            product = product.iloc[0]
            image_filename = f"{test_product_id}.jpg"
            image_path = f"Images/{image_filename}"
            
            # Vérifier si l'image existe
            if os.path.exists(image_path):
                # Nettoyer la description (enlever les \n et \t)
                description = product['description'] if pd.notna(product['description']) else product['product_name']
                if description:
                    description = description.replace('\n', ' ').replace('\t', ' ').strip()
                    # Garder seulement les 2 premières phrases pour la lisibilité
                    sentences = description.split('. ')
                    if len(sentences) > 2:
                        description = '. '.join(sentences[:2]) + '.'
                
                # Nettoyer les spécifications (parser le format Ruby/JSON)
                specs = product['product_specifications'] if pd.notna(product['product_specifications']) else f"Prix: {product['retail_price']} INR"
                if specs and specs.startswith('{"product_specification"'):
                    try:
                        # Remplacer => par : pour convertir en JSON valide
                        json_specs = specs.replace('=>', ':')
                        specs_data = json.loads(json_specs)
                        if 'product_specification' in specs_data:
                            key_specs = []
                            for spec in specs_data['product_specification'][:5]:  # Limiter à 5 specs
                                if 'key' in spec and 'value' in spec:
                                    key_specs.append(f"{spec['key']}: {spec['value']}")
                            specs = '; '.join(key_specs) if key_specs else f"Prix: {product['retail_price']} INR"
                    except:
                        specs = f"Prix: {product['retail_price']} INR"
                
                return {
                    'name': product['product_name'],
                    'description': description,
                    'specifications': specs,
                    'image_path': image_path,
                    'image_filename': image_filename
                }
            else:
                st.warning(f"⚠️ Image non trouvée: {image_path}")
                return None
        else:
            st.warning("⚠️ Produit de test non trouvé dans les données")
            return None
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du produit de test: {str(e)}")
        return None

# Charger le produit de test par défaut
default_product = load_default_test_product()

# Lancer automatiquement la prédiction sur le produit de test au premier chargement
if default_product and not st.session_state.get('auto_prediction_done', False):
    st.session_state['auto_prediction_done'] = True
    st.session_state['test_prediction_launched'] = True

# Interface de prédiction
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload de l'image")
    uploaded_file = st.file_uploader(
        "Choisissez une image de produit",
        type=['png', 'jpg', 'jpeg'],
        help="Formats supportés : PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file)
        st.image(image, caption="Image uploadée", use_column_width=True)
        
        # Informations sur l'image
        st.info(f"📏 Dimensions : {image.size[0]} x {image.size[1]} pixels")
    elif default_product and st.session_state.get('test_prediction_launched', False):
        # Afficher l'image du produit de test
        image = Image.open(default_product['image_path'])
        st.image(image, caption="Produit de test", use_column_width=True)
        st.info(f"📏 Dimensions : {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.subheader("📝 Informations du produit")
    
    if default_product and st.session_state.get('test_prediction_launched', False):
        # Utiliser les données du produit de test
        product_name = st.text_input(
            "Nom du produit",
            value=default_product['name'],
            placeholder="Ex: iPhone 14 Pro"
        )
        
        description = st.text_area(
            "Description du produit",
            value=default_product['description'],
            placeholder="Ex: Smartphone haut de gamme avec caméra professionnelle"
        )
        
        specifications = st.text_area(
            "Spécifications techniques",
            value=default_product['specifications'],
            placeholder="Ex: 6.1 pouces, 128GB, iOS 16"
        )
    else:
        # Interface normale
        product_name = st.text_input(
            "Nom du produit",
            placeholder="Ex: iPhone 14 Pro"
        )
        
        description = st.text_area(
            "Description du produit",
            placeholder="Ex: Smartphone haut de gamme avec caméra professionnelle"
        )
        
        specifications = st.text_area(
            "Spécifications techniques",
            placeholder="Ex: 6.1 pouces, 128GB, iOS 16"
        )
    
    # Bouton de prédiction
    if st.button("🔮 Prédire la catégorie", type="primary"):
        # Déterminer quelle image utiliser
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        elif default_product and st.session_state.get('test_prediction_launched', False):
            image = Image.open(default_product['image_path'])
        else:
            st.error("❌ Veuillez uploader une image avant de faire une prédiction")
            st.stop()
        
        with st.spinner("🔄 Analyse en cours..."):
            # Prédiction avec Azure ML
            text_description = f"{product_name} {description} {specifications}"
            result = azure_client.predict_category(image, text_description)
            
            # Affichage des résultats
            if 'predicted_category' in result:
                st.success("✅ Prédiction terminée !")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Catégorie prédite",
                        result['predicted_category']
                    )
                
                with col2:
                    confidence = result.get('confidence', 0.0)
                    st.metric(
                        "Confiance",
                        f"{confidence:.2%}"
                    )
                
                # Scores détaillés si disponibles
                if 'category_scores' in result:
                    st.subheader("📊 Scores par catégorie")
                    scores_df = pd.DataFrame(
                        list(result['category_scores'].items()),
                        columns=['Catégorie', 'Score']
                    ).sort_values('Score', ascending=False)
                    
                    st.bar_chart(scores_df.set_index('Catégorie'))
                    st.dataframe(scores_df)
                
                # Génération de la heatmap d'attention ONNX
                st.subheader("🔥 Heatmap d'Attention ONNX")
                attention_result = azure_client.generate_attention_heatmap(image, text_description)
                
                if attention_result and 'heatmap' in attention_result:
                    st.success("✅ Heatmap d'attention générée avec succès !")
                    
                    # Afficher la heatmap
                    heatmap_data = attention_result['heatmap']
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(heatmap_data, cmap='inferno', alpha=0.7)
                    ax.set_title("Heatmap d'Attention CLIP ONNX")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
                    
                    # Informations sur les mots-clés
                    if 'keywords' in attention_result:
                        st.write("**Mots-clés analysés :**")
                        keywords = attention_result['keywords']
                        for i, keyword in enumerate(keywords[:5], 1):
                            st.write(f"{i}. {keyword}")
                else:
                    st.warning("⚠️ Impossible de générer la heatmap d'attention")
                    
            else:
                st.error(f"❌ Erreur lors de la prédiction: {result.get('error', 'Erreur inconnue')}")
                
                # Messages d'aide spécifiques selon le type d'erreur
                error_msg = result.get('error', '').lower()
                if 'timeout' in error_msg:
                    st.warning("⏱️ **Problème de timeout détecté**")
                    st.info("💡 **Solutions possibles :**")
                    st.info("• L'endpoint Azure ML n'est pas disponible ou ne répond pas")
                    st.info("• Le service est surchargé ou en maintenance")
                    st.info("• Vérifiez la configuration de l'endpoint")
                elif '503' in error_msg or 'application error' in error_msg:
                    st.warning("🚫 **Service Azure ML indisponible (503)**")
                    st.info("💡 **Solutions possibles :**")
                    st.info("• Le service Azure ML est en maintenance ou surchargé")
                    st.info("• L'application Azure a des problèmes de ressources")
                    st.info("• Contactez l'administrateur du service Azure ML")
                else:
                    st.info("💡 Vérifiez la configuration de l'API Azure ML.")

# Lancer automatiquement la prédiction sur le produit de test
if default_product and st.session_state.get('test_prediction_launched', False):
    st.markdown("---")
    st.info("🎯 **Prédiction automatique sur le produit de test**")
    
    # Lancer la prédiction automatiquement
    if st.button("🚀 Lancer la prédiction automatique", type="primary"):
        with st.spinner("🔄 Analyse automatique en cours..."):
            # Charger l'image du produit de test
            image = Image.open(default_product['image_path'])
            
            # Prédiction avec Azure ML
            text_description = f"{default_product['name']} {default_product['description']} {default_product['specifications']}"
            result = azure_client.predict_category(image, text_description)
            
            # Affichage des résultats
            if 'predicted_category' in result:
                st.success("✅ Prédiction automatique terminée !")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Catégorie prédite",
                        result['predicted_category']
                    )
                
                with col2:
                    confidence = result.get('confidence', 0.0)
                    st.metric(
                        "Confiance",
                        f"{confidence:.2%}"
                    )
                
                # Scores détaillés si disponibles
                if 'category_scores' in result:
                    st.subheader("📊 Scores par catégorie")
                    scores_df = pd.DataFrame(
                        list(result['category_scores'].items()),
                        columns=['Catégorie', 'Score']
                    ).sort_values('Score', ascending=False)
                    
                    st.bar_chart(scores_df.set_index('Catégorie'))
                    st.dataframe(scores_df)
                
                # Génération de la heatmap d'attention ONNX
                st.subheader("🔥 Heatmap d'Attention ONNX")
                attention_result = azure_client.generate_attention_heatmap(image, text_description)
                
                if attention_result and 'heatmap' in attention_result:
                    st.success("✅ Heatmap d'attention générée avec succès !")
                    
                    # Afficher la heatmap
                    heatmap_data = attention_result['heatmap']
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(heatmap_data, cmap='inferno', alpha=0.7)
                    ax.set_title("Heatmap d'Attention CLIP ONNX")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
                    
                    # Informations sur les mots-clés
                    if 'keywords' in attention_result:
                        st.write("**Mots-clés analysés :**")
                        keywords = attention_result['keywords']
                        for i, keyword in enumerate(keywords[:5], 1):
                            st.write(f"{i}. {keyword}")
                else:
                    st.warning("⚠️ Impossible de générer la heatmap d'attention")
                    
            else:
                st.error(f"❌ Erreur lors de la prédiction automatique: {result.get('error', 'Erreur inconnue')}")

# Informations sur le modèle
st.markdown("---")
st.success("🚀 Configuration Azure ML ONNX activée")
st.info("✅ Modèles ONNX optimisés pour des performances maximales")
st.info("""
ℹ️ **Note** : Cette application utilise des modèles CLIP ONNX déployés sur Azure ML.
Les prédictions sont effectuées via l'inférence ONNX optimisée pour des performances maximales.
""")
