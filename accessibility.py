#!/usr/bin/env python3
"""
Module de gestion des options d'accessibilité pour toutes les pages
"""

import streamlit as st
import time

def init_accessibility_state():
    """Initialise l'état d'accessibilité si nécessaire"""
    if 'accessibility' not in st.session_state:
        st.session_state.accessibility = {
            'high_contrast': False,
            'large_text': False,
            'color_blind': False
        }
    
    # Initialiser un compteur de changements pour forcer les mises à jour
    if 'accessibility_change_count' not in st.session_state:
        st.session_state.accessibility_change_count = 0

def render_accessibility_sidebar():
    """Affiche les options d'accessibilité dans la sidebar"""
    with st.sidebar:
        st.header("👁️ Options d'accessibilité")
        
        # Stocker l'état précédent pour détecter les changements
        prev_high_contrast = st.session_state.accessibility.get('high_contrast', False)
        prev_large_text = st.session_state.accessibility.get('large_text', False)
        prev_color_blind = st.session_state.accessibility.get('color_blind', False)
        
        # Toutes les options d'accessibilité au même niveau
        new_high_contrast = st.checkbox(
            "🌙 Mode contraste élevé", 
            value=st.session_state.accessibility['high_contrast'],
            help="Active un mode sombre avec un contraste élevé pour une meilleure accessibilité",
            key="high_contrast_checkbox"
        )
        
        new_large_text = st.checkbox(
            "🔍 Texte agrandi", 
            value=st.session_state.accessibility['large_text'],
            help="Augmente la taille du texte pour une meilleure lisibilité",
            key="large_text_checkbox"
        )
        
        new_color_blind = st.checkbox(
            "🎨 Mode daltonien", 
            value=st.session_state.accessibility['color_blind'],
            help="Utilise des couleurs accessibles pour les personnes daltoniennes",
            key="color_blind_checkbox"
        )
        
        # Mettre à jour l'état et forcer le rafraîchissement si nécessaire
        st.session_state.accessibility['high_contrast'] = new_high_contrast
        st.session_state.accessibility['large_text'] = new_large_text
        st.session_state.accessibility['color_blind'] = new_color_blind
        
        # Si une option d'accessibilité a changé, forcer un rafraîchissement
        if (prev_high_contrast != new_high_contrast or 
            prev_large_text != new_large_text or 
            prev_color_blind != new_color_blind):
            st.session_state.accessibility_change_count += 1
            st.rerun()
        
        # Informations sur l'accessibilité
        st.info("💡 Ces options améliorent l'accessibilité de l'application selon les recommandations WCAG 2.1 AA.")

def apply_accessibility_styles():
    """Applique les styles CSS selon les options d'accessibilité sélectionnées"""
    # S'assurer que l'état d'accessibilité est initialisé
    if 'accessibility' not in st.session_state:
        init_accessibility_state()
    
    # Ajouter un timestamp pour forcer la mise à jour des styles
    timestamp = st.session_state.get('accessibility_change_count', 0)
    
    # Appliquer les styles selon les options d'accessibilité
    high_contrast = st.session_state.accessibility.get('high_contrast', False)
    large_text = st.session_state.accessibility.get('large_text', False)
    
    if high_contrast:
        css_content = """
            <style>
            /* Mode contraste élevé - Centralisé et simple - Timestamp: """ + str(timestamp) + """ */
            .stApp {
                background-color: #0f0f23 !important;
                color: #ffffff !important;
            }
            
            .main .block-container {
                background-color: #0f0f23 !important;
                color: #ffffff !important;
            }
            
            [data-testid="stSidebar"] {
                background-color: #1a1a2e !important;
            }
            
            h1, h2, h3, h4, h5, h6, p, div, span, label {
                color: #ffffff !important;
            }
            
            /* Mode texte agrandi */
            """ + ("""
            h1 { font-size: 2.5rem !important; }
            h2 { font-size: 2rem !important; }
            h3 { font-size: 1.75rem !important; }
            h4 { font-size: 1.5rem !important; }
            h5 { font-size: 1.25rem !important; }
            h6 { font-size: 1.1rem !important; }
            p, div, span, label, .stText, .stMarkdown {
                font-size: 1.1rem !important;
                line-height: 1.6 !important;
            }
            .stSelectbox, .stTextInput, .stTextArea, .stButton {
                font-size: 1.1rem !important;
            }
            .stDataFrame, .stTable {
                font-size: 1.1rem !important;
            }
            """ if large_text else "") + """
            
            .stText, .stMarkdown {
                color: #ffffff !important;
            }
            
            .stTextInput > div > div > input,
            .stTextArea > div > div > textarea,
            .stSelectbox > div > div > select {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #000000 !important;
            }
            
            /* Champ de drag and drop - garder le style par défaut */
            .stFileUploader > div > div,
            .stFileUploader > div > div > div,
            .stFileUploader > div > div > div > div {
                background-color: #f0f2f6 !important;
                border: 1px dashed #cccccc !important;
                color: #262730 !important;
            }
            
            .stFileUploader > div > div:hover,
            .stFileUploader > div > div > div:hover,
            .stFileUploader > div > div > div > div:hover {
                background-color: #e6e9ed !important;
                border-color: #999999 !important;
            }
            
            .stFileUploader .uploadedFile,
            .stFileUploader .uploadedFile > div,
            .stFileUploader .uploadedFile > div > div {
                background-color: #ffffff !important;
                color: #262730 !important;
                border: 1px solid #cccccc !important;
            }
            
            /* Texte dans le drag and drop */
            .stFileUploader p,
            .stFileUploader span,
            .stFileUploader div,
            .stFileUploader label {
                color: #262730 !important;
            }
            
            /* Bouton de sélection de fichier */
            .stFileUploader button,
            .stFileUploader button > div,
            .stFileUploader button > div > div {
                background-color: #ffffff !important;
                color: #262730 !important;
                border: 1px solid #cccccc !important;
            }
            
            .stDataFrame {
                background-color: #1a1a2e !important;
                color: #ffffff !important;
            }
            
            .stButton > button {
                background-color: #4a4a6a !important;
                color: #ffffff !important;
            }
            
            /* Graphiques Plotly */
            .js-plotly-plot {
                background: transparent !important;
            }
            
            .js-plotly-plot .plotly .main-svg {
                background: transparent !important;
            }
            
            .js-plotly-plot .plotly .main-svg text {
                fill: #ffffff !important;
            }
            
            .js-plotly-plot .plotly .main-svg .legend text {
                fill: #ffffff !important;
            }
            
            .js-plotly-plot .plotly .main-svg .gtitle text {
                fill: #ffffff !important;
            }
            
            .js-plotly-plot .plotly .main-svg .xtitle text,
            .js-plotly-plot .plotly .main-svg .ytitle text {
                fill: #ffffff !important;
            }
            
            /* Tooltips et étiquettes Plotly - Styles plus agressifs */
            .plotly .hovertext,
            .plotly .hovertext * {
                fill: #000000 !important;
                color: #000000 !important;
            }
            
            .plotly .hoverlabel,
            .plotly .hoverlabel * {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 2px solid #000000 !important;
            }
            
            .plotly .hoverlabel .hovertext,
            .plotly .hoverlabel .hovertext * {
                color: #000000 !important;
                fill: #000000 !important;
            }
            
            /* Tooltips spécifiques pour différents types de graphiques */
            .plotly .hoverlabel .name,
            .plotly .hoverlabel .value,
            .plotly .hoverlabel .text {
                color: #000000 !important;
                fill: #000000 !important;
            }
            
            /* Étiquettes de données */
            .plotly .textpoint {
                fill: #ffffff !important;
            }
            
            .plotly .textpoint text {
                fill: #ffffff !important;
            }
            
            /* Légendes */
            .plotly .legend {
                background-color: rgba(0, 0, 0, 0.8) !important;
            }
            
            .plotly .legend .legendtext {
                fill: #ffffff !important;
            }
            
            /* Forcer les styles sur tous les éléments de tooltip */
            .plotly .hoverlabel div,
            .plotly .hoverlabel span,
            .plotly .hoverlabel p {
                color: #000000 !important;
                background-color: #ffffff !important;
            }
            
            /* Styles spécifiques pour les tooltips de barres */
            .plotly .hoverlabel .extra,
            .plotly .hoverlabel .extra * {
                color: #000000 !important;
                background-color: #ffffff !important;
            }
            
            /* Graphiques Matplotlib */
            .stPyplot {
                background: transparent !important;
            }
            
            .stPyplot svg {
                background: #000000 !important;
            }
            
            .stPyplot svg text {
                fill: #ffffff !important;
            }
            
            /* Captions et légendes */
            .stCaption, .stImage caption, figcaption {
                background: transparent !important;
                color: #ffffff !important;
            }
            
            /* Images */
            .stImage {
                background: transparent !important;
            }
            
            /* Tables et DataFrames - Mode contraste élevé corrigé */
            .stDataFrame, .stTable {
                background-color: #0f0f23 !important;
                border: none !important;
            }
            
            /* Conteneur principal des tableaux */
            div[data-testid="stDataFrame"], 
            div[data-testid="stTable"] {
                background-color: #0f0f23 !important;
                border: none !important;
            }
            
            /* Tables internes */
            .stDataFrame table, .stTable table {
                background-color: #0f0f23 !important;
                color: #ffffff !important;
                border-collapse: collapse !important;
                width: 100% !important;
            }
            
            /* En-têtes de colonnes */
            .stDataFrame th, .stTable th,
            .stDataFrame .dataframe th,
            div[data-testid="stDataFrame"] th {
                background-color: #1a1a2e !important;
                color: #ffffff !important;
                border: none !important;
                padding: 8px 12px !important;
                font-weight: bold !important;
            }
            
            /* Cellules de données */
            .stDataFrame td, .stTable td,
            .stDataFrame .dataframe td,
            div[data-testid="stDataFrame"] td {
                background-color: #0f0f23 !important;
                color: #ffffff !important;
                border: none !important;
                padding: 8px 12px !important;
            }
            
            /* Lignes alternées pour meilleure lisibilité */
            .stDataFrame tr:nth-child(even) td,
            .stDataFrame .dataframe tr:nth-child(even) td {
                background-color: #1a1a2e !important;
            }
            
            .stDataFrame tr:nth-child(odd) td,
            .stDataFrame .dataframe tr:nth-child(odd) td {
                background-color: #0f0f23 !important;
            }
            
            /* Effet hover */
            .stDataFrame tr:hover td,
            .stDataFrame .dataframe tr:hover td {
                background-color: #2a2a3e !important;
            }
            
            /* Conteneur de scroll */
            .stDataFrame .dataframe-container {
                background-color: #0f0f23 !important;
                border: none !important;
            }
            
            /* Headers de colonnes spécifiques */
            .stDataFrame .column-header {
                background-color: #1a1a2e !important;
                color: #ffffff !important;
                border: none !important;
            }
            
            /* Texte dans les cellules - spécifique pour éviter les conflits */
            .stDataFrame td span,
            .stDataFrame th span,
            .stDataFrame .dataframe td span,
            .stDataFrame .dataframe th span {
                color: #ffffff !important;
            }
            
            /* Nombres et valeurs dans les cellules */
            .stDataFrame td,
            .stDataFrame th {
                color: #ffffff !important;
            }
            </style>
        """
        
        # Script JavaScript pour forcer les styles des tooltips
        js_content = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Observer pour détecter les nouveaux graphiques Plotly
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList') {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === 1 && node.classList && node.classList.contains('js-plotly-plot')) {
                                // Forcer les styles des tooltips
                                setTimeout(function() {
                                    const hoverlabels = node.querySelectorAll('.hoverlabel');
                                    hoverlabels.forEach(function(label) {
                                        label.style.backgroundColor = 'white';
                                        label.style.color = 'black';
                                        label.style.border = '2px solid black';
                                        const textElements = label.querySelectorAll('*');
                                        textElements.forEach(function(el) {
                                            el.style.color = 'black';
                                            el.style.backgroundColor = 'white';
                                        });
                                    });
                                }, 100);
                            }
                        });
                    }
                });
            });
            
            // Observer le body pour détecter les nouveaux éléments
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
        </script>
        """
        st.markdown(js_content, unsafe_allow_html=True)
        st.markdown(css_content, unsafe_allow_html=True)
    else:
        # Mode clair
        css_content_light = """
            <style>
        /* ===== MODE CLAIR ===== - Timestamp: """ + str(timestamp) + """ */
        
        .stApp {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        
        /* Mode texte agrandi */
        """ + ("""
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 2rem !important; }
        h3 { font-size: 1.75rem !important; }
        h4 { font-size: 1.5rem !important; }
        h5 { font-size: 1.25rem !important; }
        h6 { font-size: 1.1rem !important; }
        p, div, span, label, .stText, .stMarkdown {
            font-size: 1.1rem !important;
            line-height: 1.6 !important;
        }
        .stSelectbox, .stTextInput, .stTextArea, .stButton {
            font-size: 1.1rem !important;
        }
        .stDataFrame, .stTable {
            font-size: 1.1rem !important;
        }
        """ if large_text else "") + """
        
        .main, 
        .main .block-container, 
        .main .block-container > div,
        .block-container,
        .block-container > div {
            background: transparent !important;
            background-color: transparent !important;
            background-image: none !important;
            color: #262730 !important;
        }
        
        [data-testid="stSidebar"], 
        [data-testid="stSidebar"] > div {
            background-color: #f8f9fa !important;
            border-right: 2px solid #dee2e6 !important;
        }
        
        h1, h2, h3, h4, h5, h6, p, div, span, label, small, caption {
            color: #262730 !important;
        }
        
        .stText, .stMarkdown, .stWrite, .stInfo, .stSuccess, .stWarning, .stError {
            background: transparent !important;
            color: #262730 !important;
        }
        
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            background-color: #ffffff !important;
            color: #262730 !important;
            border: 1px solid #dee2e6 !important;
        }
        
        .stDataFrame, .stTable {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        
        .stCaption, .stImage caption, figcaption {
            background: transparent !important;
            color: #262730 !important;
        }
        
        .stImage {
            background: transparent !important;
        }
        
        .js-plotly-plot,
        .stPyplot {
            background: transparent !important;
        }
        
        .js-plotly-plot .plotly .main-svg text {
            fill: #262730 !important;
        }
        
        .stButton > button {
            background-color: #007bff !important;
            color: #ffffff !important;
            border: 1px solid #0056b3 !important;
        }
        
        html, body {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        
        * {
            background-image: none !important;
        }
        
        .main *,
        .block-container * {
            background-image: none !important;
        }
        </style>
        """
        st.markdown(css_content_light, unsafe_allow_html=True)
