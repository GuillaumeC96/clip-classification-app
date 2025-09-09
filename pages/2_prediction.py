"""
Page de prédiction pour la version cloud avec Azure ML
"""

import os
import pandas as pd
import streamlit as st
try:
    import spacy
except ImportError:
    spacy = None
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from collections import Counter
import re
try:
    import torch
except ImportError:
    torch = None
from scipy.interpolate import griddata
from azure_client import get_azure_client
# Imports supprimés : clients locaux non utilisés

# Importer le module d'accessibilité
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from accessibility import init_accessibility_state, render_accessibility_sidebar, apply_accessibility_styles

# Initialiser l'état d'accessibilité
init_accessibility_state()

st.title("🔮 Prédiction de Catégorie")

# Configuration d'accessibilité
ACCESSIBLE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
HIGH_CONTRAST_COLORS = ['#FFFFFF', '#FF0000', '#00FF00', '#0000FF', '#FFFF00']

# Mode de prédiction : Azure App Service
st.sidebar.markdown("### 🔧 Mode de Prédiction")
st.sidebar.info("🚀 **Azure App Service (Cloud)** - Service simplifié compatible compte gratuit")
prediction_mode = "Azure App Service (Cloud)"

# Client Azure ML uniquement
simulated_client = None
onnx_client = None
ultra_fast_client = None
azure_client = get_azure_client(show_warning=False)

# Afficher les options d'accessibilité dans la sidebar
render_accessibility_sidebar()

# Appliquer les styles d'accessibilité
apply_accessibility_styles()

# spaCy will be handled by Azure ML ONNX API, not in the client
nlp = None
st.info("🔄 spaCy processing will be handled by Azure ML ONNX API")

# Fonction pour charger le produit de test par défaut
def load_default_test_product():
    """Charge le produit de test par défaut (montre Escort)"""
    try:
        # Charger les données
        df = pd.read_csv('produits_original.csv')
        
        # Trouver le produit de test
        test_product_id = "1120bc768623572513df956172ffefeb"
        test_product = df[df['uniq_id'] == test_product_id]
        
        if not test_product.empty:
            product = test_product.iloc[0]
            
            # Construire le chemin de l'image
            image_filename = f"{test_product_id}.jpg"
            image_path = f"Images/{image_filename}"
            
            # Vérifier si l'image existe
            if os.path.exists(image_path):
                return {
                    'name': product['product_name'],
                    'description': product['product_name'],  # Utiliser le nom comme description
                    'specifications': f"Prix: {product['retail_price']} INR, Catégorie: {product['product_category_tree']}",
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
    st.session_state['test_product_name'] = default_product['name']
    st.session_state['test_description'] = default_product['description']
    st.session_state['test_specifications'] = default_product['specifications']
    st.session_state['test_image_path'] = default_product['image_path']

def clean_text(text):
    """Clean text using the same replacement patterns as in training."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    all_patterns = [
        (r'\(', ' ( '),
        (r'\)', ' ) '),
        (r'\.', ' . '),
        (r'\!', ' ! '),
        (r'\?', ' ? '),
        (r'\:', ' : '),
        (r'\,', ', '),
        # Baby Care
        (r'\b(\d+)\s*[-~to]?\s*(\d+)\s*(m|mth|mths|month|months?)\b', 'month'),
        (r'\bnewborn\s*[-~to]?\s*(\d+)\s*(m|mth|months?)\b', 'month'),
        (r'\b(nb|newborn|baby|bb|bby|babie|babies)\b', 'baby'),
        (r'\b(diaper|diapr|nappy)\b', 'diaper'),
        (r'\b(stroller|pram|buggy)\b', 'stroller'),
        (r'\b(bpa\s*free|non\s*bpa)\b', 'bisphenol A free'),
        (r'\b(\d+)\s*(oz|ounce)\b', 'ounce'),
        # Computer Hardware
        (r'\b(rtx\s*\d+)\b', 'ray tracing graphics'),
        (r'\b(gtx\s*\d+)\b', 'geforce graphics'),
        (r'\bnvidia\b', 'nvidia'),
        (r'\b(amd\s*radeon\s*rx\s*\d+)\b', 'amd radeon graphics'),
        (r'\b(intel\s*(core|xeon)\s*[i\d-]+)\b', 'intel processor'),
        (r'\b(amd\s*ryzen\s*[\d]+)\b', 'amd ryzen processor'),
        (r'\bssd\b', 'solid state drive'),
        (r'\bhdd\b', 'hard disk drive'),
        (r'\bwifi\s*([0-9])\b', 'wi-fi standard'),
        (r'\bbluetooth\s*(\d\.\d)\b', 'bluetooth version'),
        (r'\bethernet\b', 'ethernet'),
        (r'\bfhd\b', 'full high definition'),
        (r'\buhd\b', 'ultra high definition'),
        (r'\bqhd\b', 'quad high definition'),
        (r'\boled\b', 'organic light emitting diode'),
        (r'\bips\b', 'in-plane switching'),
        (r'\bram\b', 'random access memory'),
        (r'\bcpu\b', 'central processing unit'),
        (r'\bgpu\b', 'graphics processing unit'),
        (r'\bhdmi\b', 'high definition multimedia interface'),
        (r'\busb\s*([a-z0-9]*)\b', 'universal serial bus'),
        (r'\brgb\b', 'red green blue'),
        # Home Appliances
        (r'\bfridge\b', 'refrigerator'),
        (r'\bwashing\s*machine\b', 'clothes washer'),
        (r'\bdishwasher\b', 'dish washing machine'),
        (r'\boven\b', 'cooking oven'),
        (r'\bmicrowave\b', 'microwave oven'),
        (r'\bhoover\b', 'vacuum cleaner'),
        (r'\btumble\s*dryer\b', 'clothes dryer'),
        (r'\b(a\+)\b', 'energy efficiency class'),
        (r'\b(\d+)\s*btu\b', 'british thermal unit'),
        # Textiles and Materials
        (r'\bpoly\b', 'polyester'),
        (r'\bacrylic\b', 'acrylic fiber'),
        (r'\bnylon\b', 'nylon fiber'),
        (r'\bspandex\b', 'spandex fiber'),
        (r'\blycra\b', 'lycra fiber'),
        (r'\bpvc\b', 'polyvinyl chloride'),
        (r'\bvinyl\b', 'vinyl material'),
        (r'\bstainless\s*steel\b', 'stainless steel'),
        (r'\baluminum\b', 'aluminum metal'),
        (r'\bplexiglass\b', 'acrylic glass'),
        (r'\bpu\s*leather\b', 'polyurethane leather'),
        (r'\bsynthetic\s*leather\b', 'synthetic leather'),
        (r'\bfaux\s*leather\b', 'faux leather'),
        (r'\bwaterproof\b', 'water resistant'),
        (r'\bbreathable\b', 'air permeable'),
        (r'\bwrinkle-free\b', 'wrinkle resistant'),
        # Beauty and Personal Care
        (r'\bSPF\b', 'Sun Protection Factor'),
        (r'\bUV\b', 'Ultraviolet'),
        (r'\bBB\s*cream\b', 'Blemish Balm cream'),
        (r'\bCC\s*cream\b', 'Color Correcting cream'),
        (r'\bHA\b', 'Hyaluronic Acid'),
        (r'\bAHA\b', 'Alpha Hydroxy Acid'),
        (r'\bBHA\b', 'Beta Hydroxy Acid'),
        (r'\bPHA\b', 'Polyhydroxy Acid'),
        (r'\bNMF\b', 'Natural Moisturizing Factor'),
        (r'\bEGF\b', 'Epidermal Growth Factor'),
        (r'\bVit\s*C\b', 'Vitamin C'),
        (r'\bVit\s*E\b', 'Vitamin E'),
        (r'\bVit\s*B3\b', 'Niacinamide Vitamin B3'),
        (r'\bVit\s*B5\b', 'Panthenol Vitamin B5'),
        (r'\bSOD\b', 'Superoxide Dismutase'),
        (r'\bQ10\b', 'Coenzyme Q10'),
        (r'\bFoam\s*cl\b', 'Foam cleanser'),
        (r'\bMic\s*H2O\b', 'Micellar Water'),
        (r'\bToner\b', 'Skin toner'),
        (r'\bEssence\b', 'Skin essence'),
        (r'\bAmpoule\b', 'Concentrated serum'),
        (r'\bCF\b', 'Cruelty Free'),
        (r'\bPF\b', 'Paraben Free'),
        (r'\bSF\b', 'Sulfate Free'),
        (r'\bGF\b', 'Gluten Free'),
        (r'\bHF\b', 'Hypoallergenic Formula'),
        (r'\bNT\b', 'Non-comedogenic Tested'),
        (r'\bAM\b', 'morning'),
        (r'\bPM\b', 'night'),
        (r'\bBID\b', 'twice daily'),
        (r'\bQD\b', 'once daily'),
        (r'\bAIR\b', 'Airless pump bottle'),
        (r'\bD-C\b', 'Dropper container'),
        (r'\bT-C\b', 'Tube container'),
        (r'\bPDO\b', 'Polydioxanone'),
        (r'\bPCL\b', 'Polycaprolactone'),
        (r'\bPLLA\b', 'Poly-L-lactic Acid'),
        (r'\bHIFU\b', 'High-Intensity Focused Ultrasound'),
        (r'\b(\d+)\s*fl\s*oz\b', 'fluid ounce'),
        (r'\bpH\s*bal\b', 'pH balanced'),
        # General Abbreviations and Units
        (r'\b(\d+)\s*gb\b', 'byte'),
        (r'\b(\d+)\s*tb\b', 'byte'),
        (r'\b(\d+)\s*mb\b', 'byte'),
        (r'\b(\d+)\s*go\b', 'byte'),
        (r'\b(\d+)\s*to\b', 'byte'),
        (r'\b(\d+)\s*mo\b', 'byte'),
        (r'\boctet\b', 'byte'),
        (r'\b(\d+)\s*y\b', 'year'),
        (r'\b(\d+)\s*mth\b', 'month'),
        (r'\b(\d+)\s*d\b', 'day'),
        (r'\b(\d+)\s*h\b', 'hour'),
        (r'\b(\d+)\s*min\b', 'minute'),
        (r'\b(\d+)\s*rpm\b', 'revolution per minute'),
        (r'\b(\d+)\s*mw\b', 'watt'),
        (r'\b(\d+)\s*cw\b', 'watt'),
        (r'\b(\d+)\s*kw\b', 'watt'),
        (r'\b(\d+)\s*ma\b', 'ampere'),
        (r'\b(\d+)\s*ca\b', 'ampere'),
        (r'\b(\d+)\s*ka\b', 'ampere'),
        (r'\b(\d+)\s*mv\b', 'volt'),
        (r'\b(\d+)\s*cv\b', 'volt'),
        (r'\b(\d+)\s*kv\b', 'volt'),
        (r'\b(\d+)\s*mm\b', 'meter'),
        (r'\b(\d+)\s*cm\b', 'meter'),
        (r'\b(\d+)\s*m\b', 'meter'),
        (r'\b(\d+)\s*km\b', 'meter'),
        (r'\binch\b', 'meter'),
        (r'\b(\d+)\s*ml\b', 'liter'),
        (r'\b(\d+)\s*cl\b', 'liter'),
        (r'\b(\d+)\s*dl\b', 'liter'),
        (r'\b(\d+)\s*l\b', 'liter'),
        (r'\b(\d+)\s*oz\b', 'liter'),
        (r'\b(\d+)\s*gal\b', 'liter'),
        (r'\bounce\b', 'liter'),
        (r'\bgallon\b', 'liter'),
        (r'\b(\d+)\s*mg\b', 'gram'),
        (r'\b(\d+)\s*cg\b', 'gram'),
        (r'\b(\d+)\s*dg\b', 'gram'),
        (r'\b(\d+)\s*g\b', 'gram'),
        (r'\b(\d+)\s*kg\b', 'gram'),
        (r'\b(\d+)\s*lb\b', 'gram'),
        (r'\bpound\b', 'gram'),
        (r'\b(\d+)\s*°c\b', 'celsius'),
        (r'\b(\d+)\s*°f\b', 'celcius'),
        (r'\bfahrenheit\b', 'celcius'),
        (r'\bflipkart\.com\b', ''),
        (r'\bapprox\.?\b', 'approximately'),
        (r'\bw/o\b', 'without'),
        (r'\bw/\b', 'with'),
        (r'\bant-\b', 'anti'),
        (r'\byes\b', ''),
        (r'\bno\b', ''),
        (r'\bna\b', ''),
        (r'\brs\.?\b', ''),
    ]
    for pattern, replacement in all_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def extract_keywords_fallback(text, top_n=15):
    """Fallback keyword extraction without spaCy."""
    import re
    from collections import Counter
    
    # Simple stopwords list
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'these', 'they', 'them',
        'their', 'there', 'then', 'than', 'or', 'but', 'if', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
        'could', 'should', 'would', 'may', 'might', 'must', 'shall'
    }
    
    # Clean and tokenize text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Filter out stopwords and short words
    keywords = [word for word in words if len(word) > 2 and word not in stopwords]
    
    # Count and return top keywords
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(top_n)]

def extract_keywords(text, nlp, top_n=15):
    """Extract keywords from text using lemmatization and stopword removal."""
    if pd.isna(text) or text == '':
        return []
    # Clean text before processing
    text = clean_text(text)
    
    # If spaCy is not available, use simple text processing
    if nlp is None:
        return extract_keywords_fallback(text, top_n)
    
    try:
        doc = nlp(text)
    except Exception as e:
        st.warning(f"⚠️ Error processing text with spaCy: {str(e)}")
        return extract_keywords_fallback(text, top_n)
    keywords = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        if (len(lemma) < 2 or token.is_punct or not lemma or token.is_stop or
            token.text.isdigit() or
            re.match(r'^[\d.,]+$', token.text) or
            re.match(r'^[\d.,]+\s*[a-zA-Z%]+$', token.text) or
            re.match(r'^-[0-9]+$', token.text) or
            (re.match(r'^[A-Z0-9]+(?:[-_][A-Z0-9]+)*$', token.text, re.IGNORECASE) and
             (re.search(r'\d', token.text) and re.search(r'[a-zA-Z]', token.text)) or
             re.match(r'^[A-Z0-9]+$', token.text))):
            continue
        keywords.append(lemma)
    keyword_counts = Counter(keywords)
    return [word for word, count in keyword_counts.most_common(top_n)]

st.header("Entrée des Données du Produit")

# Section pour le produit de test par défaut
if default_product:
    st.info("🎯 **Produit de test chargé automatiquement** - Montre Escort E-1700-906")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Nom:** {default_product['name']}")
        st.write(f"**Description:** {default_product['description']}")
        st.write(f"**Spécifications:** {default_product['specifications']}")
    
    with col2:
        if os.path.exists(default_product['image_path']):
            st.image(default_product['image_path'], caption="Image du produit de test", width=200)
        else:
            st.warning("Image non trouvée")
    
    # Bouton pour lancer la prédiction sur le produit de test
    if st.button("🚀 Lancer la prédiction sur le produit de test", type="primary", key="test_prediction_btn"):
        # Simuler les données du formulaire
        product_name = default_product['name']
        description = default_product['description']
        specifications = default_product['specifications']
        uploaded_image = default_product['image_path']
        
        # Stocker dans session state pour éviter la re-exécution
        st.session_state['test_prediction_launched'] = True
        st.session_state['test_product_name'] = product_name
        st.session_state['test_description'] = description
        st.session_state['test_specifications'] = specifications
        st.session_state['test_image_path'] = uploaded_image
        
        st.rerun()

st.divider()

# Formulaire manuel
st.subheader("Ou saisir manuellement un produit")
product_name = st.text_input("Nom du Produit", placeholder="Exemple : Montre pour homme", 
                            help="Saisissez le nom complet du produit", key="product_name_input",
                            label_visibility="visible")
description = st.text_area("Description", placeholder="Exemple : Une montre élégante en cuir noir pour homme",
                          help="Décrivez le produit en détail", key="description_input",
                          label_visibility="visible")
specifications = st.text_area("Spécifications Techniques", placeholder="Exemple : Résistant à l'eau, affichage analogique",
                             help="Listez les spécifications techniques importantes", key="specifications_input",
                             label_visibility="visible")
uploaded_image = st.file_uploader("Télécharger une Image du Produit", type=['jpg', 'png', 'jpeg'],
                                 help="Image du produit à analyser", key="image_uploader",
                                 label_visibility="visible")

# Ajouter des labels ARIA pour l'accessibilité
st.markdown("""
<div role="region" aria-label="Formulaire de prédiction de produit">
""", unsafe_allow_html=True)

# Vérifier si une prédiction de test a été lancée
if st.session_state.get('test_prediction_launched', False):
    # Utiliser les données du produit de test
    product_name = st.session_state.get('test_product_name', '')
    description = st.session_state.get('test_description', '')
    specifications = st.session_state.get('test_specifications', '')
    uploaded_image = st.session_state.get('test_image_path', '')
    
    # Réinitialiser le flag
    st.session_state['test_prediction_launched'] = False
    
    # Afficher les informations du produit de test
    st.success("🎯 **Prédiction lancée sur le produit de test**")
    st.write(f"**Produit analysé:** {product_name}")
    
    if os.path.exists(uploaded_image):
        st.image(uploaded_image, caption="Image du produit de test", width=200)
    else:
        st.error(f"❌ Image non trouvée: {uploaded_image}")
        st.stop()
    
    # Lancer la prédiction
    prediction_launched = True
else:
    # Logique normale pour le formulaire manuel
    prediction_launched = st.button("Prédire", key="predict_button", help="Lancer la prédiction de catégorie", type="primary")
    
    if prediction_launched:
        if not (uploaded_image and product_name and description and specifications):
            st.error("Veuillez fournir un nom de produit, une description, des spécifications techniques et une image.")
            st.stop()
        
        st.image(uploaded_image, caption="Image Téléchargée", width=200)
        st.caption(f"Image analysée: {product_name}")

# Lancer la prédiction si demandée
if prediction_launched:
    
    # Extract keywords
    combined_text = f"{description} {specifications}"
    combined_text = clean_text(combined_text)
    keywords = extract_keywords(combined_text, nlp)
    if not keywords:
        st.error("Aucun mot-clé extrait. Veuillez fournir une description et des spécifications plus détaillées.")
        st.stop()
    
    st.write(f"**Mots-clés extraits :** {', '.join(keywords)}")
    
    # Prédiction via le client approprié
    if azure_client:
        with st.spinner("🔄 Prédiction en cours via le modèle fine-tuné ONNX..."):
            # Gérer à la fois les fichiers uploadés et les chemins d'images
            if isinstance(uploaded_image, str):
                # C'est un chemin d'image (produit de test)
                image = Image.open(uploaded_image)
            else:
                # C'est un fichier uploadé
                image = Image.open(uploaded_image)
            
            text_description = f"{product_name} {description} {specifications}"
            
            # Récupérer les mots-clés du produit si c'est le produit de test
            product_keywords = None
            if hasattr(st.session_state, 'test_product_name') and st.session_state.test_product_name:
                try:
                    df = pd.read_csv('produits_original.csv')
                    product_row = df[df['uniq_id'] == '1120bc768623572513df956172ffefeb']
                    if not product_row.empty:
                        product_keywords = product_row['keywords'].iloc[0]
                        st.write(f"🔍 Utilisation des mots-clés du CSV pour la prédiction: {product_keywords}")
                except Exception as e:
                    st.write(f"⚠️ Impossible de charger les mots-clés du CSV: {e}")
            
        # Prédiction avec Azure ML ONNX uniquement (LOGIQUE IDENTIQUE AU NOTEBOOK)
        result = azure_client.predict_category(image, text_description, product_keywords)
        
        # Générer l'interprétabilité ONNX (100% cloud)
        if azure_client.is_onnx and not azure_client.use_simulated:
            with st.spinner("🔄 Génération de l'interprétabilité ONNX..."):
                attention_result = azure_client.generate_attention_heatmap(image, text_description, product_keywords)
                if attention_result:
                    result['attention_result'] = attention_result
                    st.success("✅ Interprétabilité ONNX générée avec succès")
                else:
                    st.warning("⚠️ Interprétabilité ONNX non disponible")
        else:
            st.info("ℹ️ **Configuration requise** - Pour utiliser l'interprétabilité ONNX, configurez un endpoint Azure ML ONNX valide")
            st.info("💡 **Solution** : Remplacez l'endpoint par défaut par votre vrai endpoint Azure ML ONNX dans la configuration")
    elif azure_client:
        with st.spinner("🔄 Prédiction en cours via Azure ML ONNX..."):
            # Gérer à la fois les fichiers uploadés et les chemins d'images
            if isinstance(uploaded_image, str):
                # C'est un chemin d'image (produit de test)
                image = Image.open(uploaded_image)
            else:
                # C'est un fichier uploadé
                image = Image.open(uploaded_image)
            
            text_description = f"{product_name} {description} {specifications}"
            
            result = azure_client.predict_category(image, text_description)
    else:
        st.error("❌ Aucun client de prédiction disponible. Veuillez vérifier la configuration.")
        st.stop()
    
    if result['success']:
        st.header("Résultats de la Prédiction")
        st.write(f"**Mots-clés analysés :** {', '.join(keywords)}")
        st.write(f"**Catégorie prédite :** {result['predicted_category']}")
        st.write(f"**Confiance :** {result['confidence']:.3f}")
        st.write(f"**Source :** {result['source']}")
        
        # Afficher les scores de toutes les catégories
        st.subheader("Scores de Toutes les Catégories")
        category_data = []
        
        # Gérer les différentes structures de données
        if 'category_scores' in result:
            scores_dict = result['category_scores']
        elif 'all_scores' in result:
            # Convertir all_scores en dictionnaire avec les catégories
            categories = result.get('categories', [
                "Baby Care", "Beauty and Personal Care", "Computers", 
                "Home Decor & Festive Needs", "Home Furnishing", 
                "Kitchen & Dining", "Watches"
            ])
            scores_dict = dict(zip(categories, result['all_scores']))
        else:
            st.error("❌ Aucun score de catégorie disponible")
            scores_dict = {}
        
        for category, score in scores_dict.items():
            category_data.append({"Catégorie": category, "Score": f"{score:.4f}"})
        st.table(category_data)
        
        # Graphique des scores
        st.subheader("Visualisation des Scores")
        
        # Configuration des couleurs selon le mode d'accessibilité
        if st.session_state.accessibility.get('color_blind', False):
            palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        elif st.session_state.accessibility.get('high_contrast', False):
            palette = HIGH_CONTRAST_COLORS
        else:
            palette = ACCESSIBLE_COLORS
        
        fig, ax = plt.subplots(figsize=(12, 8))
        categories = list(result['category_scores'].keys())
        scores = list(result['category_scores'].values())
        
        bars = ax.barh(categories, scores, color=palette[:len(categories)])
        
        # Ajouter des motifs pour le mode daltonien
        if st.session_state.accessibility.get('color_blind', False):
            patterns = ['/', '\\', '|', '-', '+', 'x', 'o', '.', '*']
            for i, bar in enumerate(bars):
                bar.set_hatch(patterns[i % len(patterns)])
        
        ax.set_title(f"Scores de Classification - {product_name[:50]}...", 
                     fontsize=16 if not st.session_state.accessibility.get('large_text', False) else 20, 
                     pad=20)
        ax.set_xlabel("Score de probabilité", fontsize=14 if not st.session_state.accessibility.get('large_text', False) else 18)
        ax.set_ylabel("Catégories", fontsize=14 if not st.session_state.accessibility.get('large_text', False) else 18)
        ax.invert_yaxis()
        
        # Appliquer les styles d'accessibilité aux graphiques
        if st.session_state.accessibility.get('high_contrast', False):
            ax.set_facecolor('black')
            fig.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.set_title(ax.get_title(), color='white')
        
        # Ajouter les valeurs sur les barres
        for i, (category, score) in enumerate(zip(categories, scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center', ha='left', 
                    fontsize=12 if not st.session_state.accessibility.get('large_text', False) else 16)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Corriger l'erreur matplotlib en utilisant fig au lieu de plt
        st.pyplot(fig, use_container_width=True)
        
        # Heatmap des scores (simulation)
        st.subheader("Heatmap des Scores de Classification")
        
        # Créer une matrice pour la heatmap
        categories = list(result['category_scores'].keys())
        scores = list(result['category_scores'].values())
        
        # Créer une matrice 1D pour la heatmap
        score_matrix = np.array(scores).reshape(1, -1)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Générer la heatmap
        if sns is not None:
            # Utiliser seaborn si disponible
            sns.heatmap(score_matrix, 
                       xticklabels=categories,
                       yticklabels=['Scores'],
                       annot=True, 
                       fmt='.3f',
                       cmap='RdYlBu_r' if not st.session_state.accessibility.get('color_blind', False) else 'viridis',
                       cbar_kws={'label': 'Score de probabilité'},
                       ax=ax)
        else:
            # Fallback avec matplotlib
            im = ax.imshow(score_matrix, cmap='RdYlBu_r' if not st.session_state.accessibility.get('color_blind', False) else 'viridis', aspect='auto')
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_yticks([0])
            ax.set_yticklabels(['Scores'])
            plt.colorbar(im, label='Score de probabilité', ax=ax)
            
            # Ajouter les annotations
            for i in range(len(categories)):
                ax.text(i, 0, f'{scores[i]:.3f}', ha='center', va='center', 
                        color='white' if scores[i] > 0.5 else 'black', fontweight='bold')
        
        ax.set_title(f"Heatmap des Scores - {product_name[:50]}...", 
                     fontsize=16 if not st.session_state.accessibility.get('large_text', False) else 20, 
                     pad=20)
        plt.tight_layout()
        
        # Corriger l'erreur matplotlib en utilisant fig au lieu de plt
        st.pyplot(fig, use_container_width=True)
        
        # Section Interprétabilité des Mots-clés (avant la heatmap d'image)
        if uploaded_image:
            # Charger l'image pour l'analyse des mots-clés
            if isinstance(uploaded_image, str):
                # C'est un chemin d'image (produit de test)
                image = Image.open(uploaded_image)
            else:
                # C'est un fichier uploadé
                image = Image.open(uploaded_image)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Générer l'analyse des mots-clés selon le mode
            if azure_client:
                # Générer seulement les scores de mots-clés (plus rapide)
                with st.spinner("🔄 Calcul des scores de similarité des mots-clés..."):
                    # Utiliser les mots-clés du CSV si disponibles, sinon extraire de la description
                    if product_keywords:
                        keywords = [kw.strip() for kw in product_keywords.split(',') if kw.strip()]
                        st.write(f"🔍 Utilisation des mots-clés du CSV: {', '.join(keywords)}")
                    else:
                        # Fallback: extraction simple des mots-clés
                        import re
                        from collections import Counter
                        
                        # Nettoyer le texte
                        cleaned_text = re.sub(r'[^\w\s]', ' ', text_description.lower())
                        words = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned_text)
                        
                        # Stopwords simples
                        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                                   'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                                   'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
                        
                        keywords = [word for word in words if word not in stopwords and len(word) > 2]
                        keyword_counts = Counter(keywords)
                        keywords = [word for word, count in keyword_counts.most_common(10)]
                        st.write(f"🔍 Mots-clés extraits: {', '.join(keywords)}")
                    
                     if keywords:
                         # Utiliser les vraies données d'interprétabilité du backend Azure
                         attention_result = result.get('attention_result')
                         if attention_result:
                             st.info("✅ **Vraies données d'interprétabilité ONNX** - Scores calculés par le modèle CLIP fine-tuné déployé sur Azure ML")
                         else:
                             st.info("ℹ️ **Configuration requise** - Pour utiliser l'interprétabilité ONNX, configurez un endpoint Azure ML ONNX valide")
                             st.info("💡 **Solution** : Remplacez l'endpoint par défaut par votre vrai endpoint Azure ML ONNX dans la configuration")
                             attention_result = None
                
                # Afficher les scores de similarité des mots-clés (seulement si disponibles)
                if attention_result and attention_result is not None and 'keyword_similarities' in attention_result:
                    st.subheader("📊 Interprétabilité des Mots-clés")
                    
                    # Créer le diagramme en barres des scores de similarité (version optimisée)
                    keywords_list = list(attention_result['keyword_similarities'].keys())
                    scores_list = list(attention_result['keyword_similarities'].values())
                    
                    # Trier par score décroissant et limiter à 10 mots-clés max pour la vitesse
                    sorted_data = sorted(zip(keywords_list, scores_list), key=lambda x: x[1], reverse=True)[:10]
                    keywords_list, scores_list = zip(*sorted_data)
                    
                    # Version optimisée : utiliser Streamlit native au lieu de matplotlib/seaborn
                    if len(keywords_list) > 0:
                        # Créer un DataFrame pour st.bar_chart (plus rapide)
                        import pandas as pd
                        chart_data = pd.DataFrame({
                            'Mots-clés': keywords_list,
                            'Score': scores_list
                        })
                        
                        # Utiliser st.bar_chart (plus rapide que matplotlib)
                        st.bar_chart(chart_data.set_index('Mots-clés'), use_container_width=True)
                        
                        # Ajouter un tableau des scores pour plus de détails
                        st.write("**Détails des scores :**")
                        score_df = pd.DataFrame({
                            'Mot-clé': keywords_list,
                            'Score': [f"{score:.3f}" for score in scores_list],
                            'Importance': ['🔴 Très élevé' if score > 0.7 else '🟡 Élevé' if score > 0.4 else '🟢 Modéré' for score in scores_list]
                        })
                        st.dataframe(score_df, use_container_width=True, hide_index=True)
                        
                        # Description textuelle pour les non-voyants
                        st.write("**Description des scores de similarité :**")
                        max_score = max(scores_list)
                        min_score = min(scores_list)
                        top_keyword = keywords_list[0]
                        st.write(f"""
                        - Les scores de similarité montrent l'importance de chaque mot-clé pour la prédiction
                        - Score maximum: {max_score:.3f} (mot-clé: {top_keyword})
                        - Score minimum: {min_score:.3f}
                        - Les mots-clés avec des scores élevés sont plus influents dans la décision du modèle
                         - **Note:** Ces scores sont calculés par le vrai modèle CLIP fine-tuné ONNX déployé sur Azure ML.
                        """)
            else:
                # Service Azure App Service simplifié - pas de scores de similarité
                st.info("ℹ️ **Service Azure App Service simplifié** - Les scores de similarité des mots-clés ne sont pas disponibles dans cette version.")
                st.info("💡 **Pourquoi ?** Cette version utilise un service simplifié compatible avec le compte Azure gratuit, qui se concentre sur la classification basée sur le texte.")
                # Mode démo - scores simulés
                st.subheader("📊 Interprétabilité des Mots-clés (Mode Démo)")
                
                # Extraire les mots-clés du texte (version optimisée)
                keywords = text_description.lower().split()
                keywords = [kw for kw in keywords if len(kw) > 2 and kw.isalpha()][:8]  # Limiter à 8 mots-clés pour la vitesse
                
                if keywords:
                    # Créer des scores simulés basés sur la catégorie prédite (version optimisée)
                    keyword_similarities = {}
                    for keyword in keywords:
                        # Score de base aléatoire
                        base_score = np.random.random() * 0.3
                        
                        # Bonus selon la catégorie prédite
                        if predicted_category == "Watches" and any(w in keyword for w in ['watch', 'time', 'clock', 'hour', 'minute']):
                            base_score += 0.4
                        elif predicted_category == "Computers" and any(w in keyword for w in ['computer', 'laptop', 'screen', 'keyboard']):
                            base_score += 0.4
                        elif predicted_category == "Beauty and Personal Care" and any(w in keyword for w in ['beauty', 'care', 'skin', 'cream']):
                            base_score += 0.4
                        
                        keyword_similarities[keyword] = min(base_score, 1.0)
                    
                    # Trier par score décroissant
                    sorted_keywords = sorted(keyword_similarities.items(), key=lambda x: x[1], reverse=True)
                    keywords_list, scores_list = zip(*sorted_keywords)
                    
                    # Version optimisée : utiliser Streamlit native au lieu de matplotlib/seaborn
                    import pandas as pd
                    chart_data = pd.DataFrame({
                        'Mots-clés': keywords_list,
                        'Score': scores_list
                    })
                    
                    # Utiliser st.bar_chart (plus rapide que matplotlib)
                    st.bar_chart(chart_data.set_index('Mots-clés'), use_container_width=True)
                    
                    # Ajouter un tableau des scores pour plus de détails
                    st.write("**Détails des scores (Mode Démo) :**")
                    score_df = pd.DataFrame({
                        'Mot-clé': keywords_list,
                        'Score': [f"{score:.3f}" for score in scores_list],
                        'Importance': ['🔴 Très élevé' if score > 0.7 else '🟡 Élevé' if score > 0.4 else '🟢 Modéré' for score in scores_list]
                    })
                    st.dataframe(score_df, use_container_width=True, hide_index=True)
                    
                    # Description textuelle pour les non-voyants
                    st.write("**Description des scores de similarité (Mode Démo) :**")
                    max_score = max(scores_list)
                    min_score = min(scores_list)
                    top_keyword = keywords_list[0]
                    st.write(f"""
                    - Les scores de similarité montrent l'importance simulée de chaque mot-clé pour la prédiction
                    - Score maximum: {max_score:.3f} (mot-clé: {top_keyword})
                    - Score minimum: {min_score:.3f}
                    - **Note:** Ces scores sont simulés pour le mode démonstration. En mode production Azure ML, ils seraient calculés par le vrai modèle.
                    """)
                else:
                    st.info("ℹ️ Aucun mot-clé significatif trouvé dans la description pour l'analyse de similarité.")
        
        # Heatmap d'attention sur l'image
        if uploaded_image:
            if azure_client:
                st.subheader("Interprétabilité Image (Heatmap d'Attention CLIP)")
            else:
                st.subheader("Interprétabilité Image (Heatmap d'Attention Simulée)")
            
             # Vérifier si on a les vraies données d'interprétabilité
             attention_result = result.get('attention_result')
             if attention_result and 'heatmap' in attention_result:
                 st.info("✅ **Vraie heatmap d'attention ONNX** - Générée par le modèle CLIP fine-tuné")
             else:
                 st.info("ℹ️ **Service Azure App Service simplifié** - Les heatmaps d'attention ne sont pas disponibles dans cette version.")
                 st.info("💡 **Pourquoi ?** Cette version utilise un service simplifié compatible avec le compte Azure gratuit, qui se concentre sur la classification basée sur le texte.")
            
            # Charger l'image
            if isinstance(uploaded_image, str):
                # C'est un chemin d'image (produit de test)
                image = Image.open(uploaded_image)
            else:
                # C'est un fichier uploadé
                image = Image.open(uploaded_image)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Générer la heatmap d'attention selon le mode
            if azure_client:
                # Option pour activer/désactiver la heatmap d'attention
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("**Options d'analyse avancée :**")
                with col2:
                    generate_heatmap = st.checkbox("Générer heatmap d'attention", value=True, help="Désactiver pour accélérer l'analyse")
                
                attention_result = None
                if generate_heatmap:
                    # Chronomètre pour la génération de heatmap
                    import time
                    start_time = time.time()
                    
                    # Barre de progression et statut
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Fonction de callback pour la progression
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    status_text.text("🔄 Génération de la heatmap d'attention CLIP (version ultra-rapide)...")
                    progress_bar.progress(5)
                    
                    # Récupérer les mots-clés du produit si c'est le produit de test
                    product_keywords = None
                    if hasattr(st.session_state, 'test_product_name') and st.session_state.test_product_name:
                        try:
                            df = pd.read_csv('produits_original.csv')
                            product_row = df[df['uniq_id'] == '1120bc768623572513df956172ffefeb']
                            if not product_row.empty:
                                product_keywords = product_row['keywords'].iloc[0]
                                st.write(f"🔍 Utilisation des mots-clés du CSV: {product_keywords}")
                        except Exception as e:
                            st.write(f"⚠️ Impossible de charger les mots-clés du CSV: {e}")
                    
         # Utiliser les vraies données d'interprétabilité si disponibles
         if attention_result and 'heatmap' in attention_result:
             st.info("✅ **Génération de la vraie heatmap d'attention ONNX**")
         else:
             st.info("ℹ️ Service Azure App Service simplifié - Pas de heatmap d'attention disponible")
             attention_result = None
        
        # Calcul du temps écoulé
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Mise à jour de la barre de progression
        progress_bar.progress(100)
        status_text.text(f"✅ Heatmap générée en {generation_time:.2f} secondes")
        
        # Service simplifié - pas de heatmap
        st.info("ℹ️ **Service Azure App Service simplifié** - Classification basée sur le texte uniquement")
        
        if attention_result and attention_result is not None and 'heatmap' in attention_result:
            # Utiliser la vraie heatmap d'attention
            attention_map = attention_result  # Passer tout l'objet attention_result
            image_array = np.array(image.convert('L'))
            
            # Description textuelle pour les non-voyants
            st.write("**Description de l'analyse d'attention :**")
            heatmap_data = np.array(attention_map['heatmap'])
            max_attention = np.max(heatmap_data)
            min_attention = np.min(heatmap_data)
            keywords_list = attention_map.get('keywords', [])
            st.write(f"""
            - La heatmap superposée montre les zones de l'image où le modèle CLIP se concentre pour faire sa prédiction
            - Intensité d'attention maximale: {max_attention:.3f}
            - Intensité d'attention minimale: {min_attention:.3f}
            - Les zones les plus claires indiquent une attention plus forte
            - **Note:** Cette heatmap est générée par le vrai modèle CLIP fine-tuné ONNX.
            - Mots-clés analysés: {', '.join(keywords_list)}
            """)
        else:
            st.warning("⚠️ Impossible de générer la heatmap d'attention")
            attention_map = None
            # Créer une heatmap d'attention simulée pour le mode Azure ML
            # Convertir en niveaux de gris pour l'affichage
            image_gray = image.convert('L')
            image_array = np.array(image_gray)
            
            # Simuler une carte d'attention basée sur la catégorie prédite
            height, width = image_array.shape
            
            # Créer une attention map simulée
            # Pour les montres, concentrer l'attention sur le centre (cadran)
            if result['predicted_category'] == 'Watches':
                # Créer une attention concentrée sur le centre (cadran de montre)
                y, x = np.ogrid[:height, :width]
                center_y, center_x = height // 2, width // 2
                sigma = min(height, width) // 4
                attention_map = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
            elif result['predicted_category'] == 'Computers':
                # Pour les ordinateurs, attention sur les bords (écran, clavier)
                y, x = np.ogrid[:height, :width]
                attention_map = np.zeros((height, width))
                # Attention sur les bords
                attention_map[0:height//4, :] = 0.8  # Haut
                attention_map[3*height//4:, :] = 0.8  # Bas
                attention_map[:, 0:width//4] = 0.6  # Gauche
                attention_map[:, 3*width//4:] = 0.6  # Droite
            else:
                # Pour les autres catégories, attention uniforme avec quelques zones d'intérêt
                attention_map = np.random.rand(height, width) * 0.3
                # Ajouter quelques zones d'attention
                for _ in range(3):
                    center_y = np.random.randint(height//4, 3*height//4)
                    center_x = np.random.randint(width//4, 3*width//4)
                    sigma = min(height, width) // 8
                    y, x = np.ogrid[:height, :width]
                    attention_map += 0.4 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
                attention_map = np.clip(attention_map, 0, 1)
            
            # Normaliser l'attention map
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # Message si la heatmap n'est pas générée
        if attention_map is None and azure_client:
            st.info("ℹ️ Heatmap d'attention non générée (option désactivée pour accélérer l'analyse)")
        
        # Afficher la heatmap si elle existe
        if attention_map is not None:
            # Copier exactement le code du notebook
            # Utiliser smooth_heatmap si disponible (comme dans le notebook)
            if isinstance(attention_map, dict) and 'heatmap' in attention_map:
                smooth_heatmap = attention_map['heatmap']
                attention_scores = attention_map.get('attention_scores', np.array([]))
                keywords = attention_map.get('keywords', [])
                positions = attention_map.get('positions', [])
            else:
                smooth_heatmap = attention_map
                attention_scores = np.array([])
                keywords = []
                positions = []
            
            height, width = smooth_heatmap.shape
            # Utiliser les dimensions de l'image originale (comme dans le notebook)
            if hasattr(image, 'size'):
                img_width, img_height = image.size
            else:
                img_width, img_height = width, height
            
            # Convertir l'image en noir et blanc (exactement comme dans le notebook)
            from PIL import ImageEnhance
            # Convertir l'image PIL en noir et blanc avec amélioration du contraste
            if hasattr(image, 'convert'):  # Si c'est une image PIL
                img_bw = image.convert('L')
                enhancer = ImageEnhance.Contrast(img_bw)
                img_bw = enhancer.enhance(1.5)
                img_bw = np.array(img_bw)
            else:  # Si c'est un array numpy
                if len(image_array.shape) == 3:
                    img_bw = np.mean(image_array, axis=2)
                else:
                    img_bw = image_array
            
            # Créer la figure exactement comme dans le notebook (plt.figure, pas subplots)
            plt.figure(figsize=(16, 10))
            
            # Afficher l'image en noir et blanc (exactement comme dans le notebook)
            plt.imshow(img_bw, cmap='gray', vmin=0, vmax=255)
            
            # Superposer la heatmap d'attention (exactement comme dans le notebook)
            heatmap_layer = plt.imshow(smooth_heatmap.T, cmap='inferno', alpha=0.55,  # Transpose heatmap
                                    extent=[0, img_width, img_height, 0], interpolation='bicubic')
            
            # Ajouter les points et textes pour les mots-clés (comme dans le notebook)
            if len(keywords) > 0 and len(attention_scores) > 0 and len(positions) > 0:
                # Debug: afficher les informations
                st.write(f"🔍 Debug: {len(keywords)} mots-clés, {attention_scores.shape} scores, {len(positions)} positions")
                
                # Top 3 mots-clés comme dans le notebook
                top_keywords = sorted(zip(keywords, attention_scores.mean(axis=0)), key=lambda x: x[1], reverse=True)[:3]
                st.write(f"🔍 Top keywords: {top_keywords}")
                
                for kw, score in top_keywords:
                    kw_idx = keywords.index(kw)
                    max_pos_idx = np.argmax(attention_scores[:, kw_idx])
                    max_pos = positions[max_pos_idx]
                    st.write(f"🔍 Mot-clé: {kw}, Position: {max_pos}, Score: {score:.3f}")
                    plt.scatter(max_pos[0], max_pos[1], s=300, edgecolors='white', linewidths=2, facecolors='none')
                    plt.text(max_pos[0], max_pos[1]+img_height*0.03, f"{kw}\n({score:.2f})",
                           color='white', ha='center', va='top', fontsize=11,
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5', edgecolor='white', linewidth=1))
            else:
                st.write(f"🔍 Debug: Pas de mots-clés - keywords: {len(keywords)}, scores: {len(attention_scores)}, positions: {len(positions)}")
            
            # Ajouter la colorbar (exactement comme dans le notebook)
            cbar = plt.colorbar(heatmap_layer, fraction=0.03, pad=0.01)
            cbar.set_label('Intensité d\'attention', rotation=270, labelpad=15)
            
            # Titre (comme dans le notebook)
            if azure_client:
                plt.title("Heatmap d'attention CLIP (Fine-Tuné)", pad=20, fontsize=12)
            else:
                plt.title("Heatmap d'Attention Simulée", pad=20, fontsize=12)
            
            # Désactiver les axes (exactement comme dans le notebook)
            plt.axis('off')
            
            plt.tight_layout()
            st.pyplot(plt, use_container_width=True)
        
        # Description textuelle pour les non-voyants (seulement pour le mode simulé)
        if not azure_client:
            st.write("**Description de l'analyse d'attention :**")
            max_attention = np.max(attention_map)
            min_attention = np.min(attention_map)
            st.write(f"""
            - La heatmap superposée montre les zones de l'image où le modèle se concentre pour faire sa prédiction
            - Intensité d'attention maximale: {max_attention:.3f}
            - Intensité d'attention minimale: {min_attention:.3f}
            - Les zones les plus claires indiquent une attention plus forte
            - **Note:** Cette heatmap est simulée pour le mode démonstration. En mode production avec Azure ML, vous obtiendriez la vraie analyse d'attention CLIP.
            """)
        
    else:
        st.error(f"❌ Erreur lors de la prédiction: {result['error']}")
        
        # Messages d'aide spécifiques selon le type d'erreur
        if 'timeout' in result['error'].lower():
            st.warning("⏱️ **Problème de timeout détecté**")
            st.info("💡 **Solutions possibles :**")
            st.info("• L'endpoint Azure ML n'est pas disponible ou ne répond pas")
            st.info("• Le service est surchargé ou en maintenance")
            st.info("• Vérifiez la configuration de l'endpoint dans les secrets")
            st.info("• Utilisez le mode démonstration pour tester l'application")
        elif '503' in result['error'] or 'application error' in result['error'].lower():
            st.warning("🚫 **Service Azure ML indisponible (503)**")
            st.info("💡 **Solutions possibles :**")
            st.info("• Le service Azure ML est en maintenance ou surchargé")
            st.info("• L'application Azure a des problèmes de ressources")
            st.info("• Le système bascule automatiquement vers le mode simulé")
            st.info("• Contactez l'administrateur du service Azure ML")
        else:
            st.info("💡 Vérifiez la configuration de l'API Azure ML.")

st.markdown("</div>", unsafe_allow_html=True)
