import streamlit as st
import pandas as pd
import plotly.express as px
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None
import matplotlib.pyplot as plt
from PIL import Image
import os
try:
    import torch
except ImportError:
    torch = None

# Configuration
KEYWORD_FREQ_PATH = 'keyword_frequencies.csv'

# Importer le module d'accessibilité
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from accessibility import init_accessibility_state, render_accessibility_sidebar, apply_accessibility_styles

# Initialiser l'état d'accessibilité
init_accessibility_state()

# Charger les données si elles ne sont pas disponibles
if 'df' not in st.session_state:
    @st.cache_data
    def load_and_process_data():
        """Charge et traite les données des produits"""
        try:
            # Charger les données
            df = pd.read_csv('produits_original.csv')
            
            # Traiter les catégories (structure différente dans produits_demo.csv)
            import ast
            df['categories'] = df['product_category_tree'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            # Extraire seulement la catégorie principale (avant le premier >>)
            df['main_category'] = df['categories'].apply(lambda x: x[0].split(' >> ')[0] if x and len(x) > 0 else 'Unknown')
            df['sub_categories'] = df['categories'].apply(lambda x: x[0].split(' >> ')[1] if x and len(x) > 0 and ' >> ' in x[0] else 'Unknown')
            
            # Ajouter des informations sur les images (colonne 'image' dans produits_original.csv)
            df['image_exists'] = df['image'].apply(lambda x: os.path.exists(f"Images/{x}") if pd.notna(x) else False)
            df['image_pixels'] = df['image'].apply(lambda x: get_image_pixels(f"Images/{x}") if pd.notna(x) and os.path.exists(f"Images/{x}") else 0)
            df['aspect_ratio'] = df['image'].apply(lambda x: get_aspect_ratio(f"Images/{x}") if pd.notna(x) and os.path.exists(f"Images/{x}") else 0)
            
            return df
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données: {str(e)}")
            return pd.DataFrame()
    
    def get_image_pixels(image_path):
        """Obtient le nombre de pixels d'une image"""
        try:
            with Image.open(image_path) as img:
                return img.width * img.height
        except:
            return 0
    
    def get_aspect_ratio(image_path):
        """Obtient le ratio d'aspect d'une image"""
        try:
            with Image.open(image_path) as img:
                return img.width / img.height
        except:
            return 0
    
    import ast
    with st.spinner("🔄 Chargement des données..."):
        st.session_state.df = load_and_process_data()


# Configuration de page supprimée - gérée par interface.py

st.title("Analyse Exploratoire des Données (EDA)")

# Access data from session state
if 'df' not in st.session_state or st.session_state.df.empty:
    st.error("❌ Aucune donnée disponible dans st.session_state.df. Veuillez vérifier le chargement des données dans interface.py.")
    st.stop()
df = st.session_state.df

# Validate DataFrame
required_columns = ['main_category', 'sub_categories', 'image', 'image_exists', 'image_pixels', 'aspect_ratio']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"❌ Colonnes manquantes dans le DataFrame : {missing_columns}")
    st.stop()

# Configuration d'accessibilité pour les graphiques
ACCESSIBLE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Configuration des couleurs selon le mode d'accessibilité
if st.session_state.accessibility.get('color_blind', False):
    PLOTLY_COLORS = px.colors.qualitative.Safe  # Accessible palette for color-blind users
elif st.session_state.accessibility.get('high_contrast', False):
    # Palette optimisée pour le mode contraste élevé (couleurs vives sur fond sombre)
    PLOTLY_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                     '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                     '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2']
else:
    PLOTLY_COLORS = ACCESSIBLE_COLORS

# Données structurées
st.subheader("Données Structurées")
st.write("**Informations de débogage :**")
st.write(f"Colonnes du DataFrame : {list(df.columns)}")
st.write(f"Nombre de lignes : {len(df)}")
st.write(f"Valeurs manquantes par colonne :")
st.dataframe(df.isna().sum())

st.write("**Statistiques descriptives (Numériques) :**")
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
if not numerical_cols.empty:
    st.dataframe(df[numerical_cols].describe())
else:
    st.warning("⚠️ Aucune colonne numérique disponible pour les statistiques.")

st.write("**Statistiques descriptives (Catégoriques) :**")
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
if not categorical_cols.empty:
    st.dataframe(df[categorical_cols].describe())
else:
    st.warning("⚠️ Aucune colonne catégorique disponible pour les statistiques.")

st.write("**Nombre de produits par catégorie principale :**")
category_count = df['main_category'].value_counts()
if category_count.empty:
    st.warning("⚠️ Aucune catégorie principale trouvée dans le DataFrame.")
else:
    st.dataframe(category_count)

    # Graphique accessible avec couleurs contrastées
    bg_color = '#000000' if st.session_state.accessibility.get('high_contrast', False) else '#FFFFFF'
    text_color = '#FFFFFF' if st.session_state.accessibility.get('high_contrast', False) else '#000000'
    
    fig1 = px.bar(category_count, x=category_count.index, y=category_count.values, 
                  title="Nombre de Produits par Catégorie Principale",
                  color=category_count.index,
                  color_discrete_sequence=PLOTLY_COLORS[:len(category_count)])
    
    fig1.update_layout(
        xaxis_title="Catégories",
        yaxis_title="Nombre de produits",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(size=14 if not st.session_state.accessibility.get('large_text', False) else 18, color=text_color),
        legend_title="Catégories",
        legend=dict(font=dict(color=text_color)),
        margin=dict(l=50, r=50, t=50, b=100),  # Ensure enough space for rotated labels
        hoverlabel=dict(
            bgcolor="white",
            font_size=14 if not st.session_state.accessibility.get('large_text', False) else 16,
            font_family="Arial, sans-serif",
            font_color="black",
            bordercolor="black"
        )
    )
    fig1.update_xaxes(
        tickangle=45,
        tickfont=dict(color=text_color, size=14 if not st.session_state.accessibility.get('large_text', False) else 16)
    )
    fig1.update_yaxes(
        tickfont=dict(color=text_color, size=14 if not st.session_state.accessibility.get('large_text', False) else 16)
    )
    st.plotly_chart(fig1, use_container_width=True, aria_label="Graphique du nombre de produits par catégorie principale")

    # Alternative textuelle pour les utilisateurs de lecteurs d'écran
    st.write("**Données textuelles du graphique :**")
    for category, count in category_count.items():
        st.write(f"- {category}: {count} produits")

st.write("**Nombre de produits par branche de catégories :**")
subcat_count = df['sub_categories'].value_counts().head(20)
if subcat_count.empty:
    st.warning("⚠️ Aucune sous-catégorie trouvée dans le DataFrame.")
else:
    st.dataframe(subcat_count)

    # Graphique en camembert avec couleurs accessibles
    fig2 = px.pie(subcat_count, values=subcat_count.values, names=subcat_count.index, 
                  title="Top 20 Branches de Catégories",
                  color_discrete_sequence=PLOTLY_COLORS)
    fig2.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Valeur: %{value}<br>Pourcentage: %{percent}',
        textfont=dict(color=text_color, size=14 if not st.session_state.accessibility.get('large_text', False) else 16)
    )
    fig2.update_layout(
        uniformtext_minsize=12 if not st.session_state.accessibility.get('large_text', False) else 16,
        uniformtext_mode='hide',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(color=text_color, size=10)),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        margin=dict(r=200),  # Ajouter une marge à droite pour la légende
        hoverlabel=dict(
            bgcolor="white",
            font_size=14 if not st.session_state.accessibility.get('large_text', False) else 16,
            font_family="Arial, sans-serif",
            font_color="black",
            bordercolor="black"
        )
    )
    st.plotly_chart(fig2, use_container_width=True, aria_label="Graphique en camembert des top 20 branches de catégories")

# Données textuelles non structurées
st.subheader("Données Textuelles Non Structurées")
try:
    keyword_freq_df = pd.read_csv(KEYWORD_FREQ_PATH)
    if keyword_freq_df.empty or 'Mot Clé' not in keyword_freq_df.columns or 'Fréquence' not in keyword_freq_df.columns:
        st.error(f"❌ Le fichier {KEYWORD_FREQ_PATH} est vide ou ne contient pas les colonnes attendues ('Mot Clé', 'Fréquence').")
    else:
        total_keywords = keyword_freq_df['Fréquence'].sum()
        st.write(f"**Nombre total de mots-clés :** {total_keywords}")
        st.write("**Fréquence des mots-clés (Top 50) :**")
        st.dataframe(keyword_freq_df.head(50))
        
        csv = keyword_freq_df.to_csv(index=False).encode('utf-8')
        
        # Empêcher le bouton de téléchargement de changer en mode contraste élevé
        if st.session_state.accessibility.get('high_contrast', False):
            st.markdown("""
            <style>
            /* Garder les styles par défaut du bouton de téléchargement */
            div[data-testid="stDownloadButton"] button {
                background-color: rgb(255, 255, 255) !important;
                color: rgb(38, 39, 48) !important;
                border: 1px solid rgba(49, 51, 63, 0.2) !important;
            }
            div[data-testid="stDownloadButton"] button:hover {
                background-color: rgba(49, 51, 63, 0.1) !important;
                color: rgb(38, 39, 48) !important;
                border: 1px solid rgba(49, 51, 63, 0.2) !important;
            }
            /* Forcer la couleur du texte à rester sombre */
            div[data-testid="stDownloadButton"] button * {
                color: rgb(38, 39, 48) !important;
            }
            div[data-testid="stDownloadButton"] button span {
                color: rgb(38, 39, 48) !important;
            }
            /* Texte blanc lors du survol */
            div[data-testid="stDownloadButton"] button:hover * {
                color: white !important;
            }
            div[data-testid="stDownloadButton"] button:hover span {
                color: white !important;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Créer un conteneur avec un ID unique pour le bouton
        download_container = st.container()
        with download_container:
            st.download_button(
                label="Télécharger les fréquences des mots clés (CSV)",
                data=csv,
                file_name="keyword_frequencies.csv",
                mime="text/csv",
                key="download_keywords_csv"
            )
        
        # Graphique barre accessible
        fig3 = px.bar(keyword_freq_df.head(50), x='Mot Clé', y='Fréquence', 
                      title="Fréquence des Mots-Clés (Top 50)",
                      color='Fréquence',
                      color_continuous_scale='viridis' if st.session_state.accessibility.get('color_blind', False) 
                      else 'plasma' if st.session_state.accessibility.get('high_contrast', False) 
                      else 'Blues')
        fig3.update_layout(
            xaxis_title="Mots-clés",
            yaxis_title="Fréquence",
            xaxis_tickangle=45,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(size=14 if not st.session_state.accessibility.get('large_text', False) else 18, color=text_color),
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white" if st.session_state.accessibility.get('high_contrast', False) else "rgba(255,255,255,0.8)",
                font_size=14 if not st.session_state.accessibility.get('large_text', False) else 16,
                font_family="Arial, sans-serif",
                font_color="black" if st.session_state.accessibility.get('high_contrast', False) else "black"
            )
        )
        fig3.update_xaxes(
            tickfont=dict(color=text_color, size=14 if not st.session_state.accessibility.get('large_text', False) else 16)
        )
        fig3.update_yaxes(
            tickfont=dict(color=text_color, size=14 if not st.session_state.accessibility.get('large_text', False) else 16)
        )
        st.plotly_chart(fig3, use_container_width=True, aria_label="Graphique en barres des fréquences des mots-clés (Top 50)")

        # Graphique camembert accessible
        fig4 = px.pie(keyword_freq_df.head(20), values='Fréquence', names='Mot Clé', 
                      title="Top 20 Mots-Clés par Fréquence",
                      color_discrete_sequence=PLOTLY_COLORS)
        fig4.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Fréquence: %{value}<br>Pourcentage: %{percent}',
            textfont=dict(color=text_color, size=14 if not st.session_state.accessibility.get('large_text', False) else 16)
        )
        fig4.update_layout(
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            legend=dict(font=dict(color=text_color)),
            hoverlabel=dict(
                bgcolor="white" if st.session_state.accessibility.get('high_contrast', False) else "rgba(255,255,255,0.8)",
                font_size=14 if not st.session_state.accessibility.get('large_text', False) else 16,
                font_family="Arial, sans-serif",
                font_color="black" if st.session_state.accessibility.get('high_contrast', False) else "black"
            )
        )
        st.plotly_chart(fig4, use_container_width=True, aria_label="Graphique en camembert des top 20 mots-clés par fréquence")

        # Nuage de mots avec contraste amélioré
        top_keywords = dict(keyword_freq_df.head(50)[['Mot Clé', 'Fréquence']].values)
        if top_keywords:
            # Choisir la palette en fonction du mode d'accessibilité
            if st.session_state.accessibility.get('color_blind', False):
                colormap = 'viridis'
            elif st.session_state.accessibility.get('high_contrast', False):
                colormap = 'hot'
            else:
                colormap = 'plasma'
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='black' if st.session_state.accessibility.get('high_contrast', False) else 'white',
                colormap=colormap,
                contour_color='white' if st.session_state.accessibility.get('high_contrast', False) else 'black',
                contour_width=1
            ).generate_from_frequencies(top_keywords)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Nuage de Mots des Mots-Clés les Plus Fréquents", 
                     fontsize=16 if not st.session_state.accessibility.get('large_text', False) else 20, 
                     pad=20,
                     color='white' if st.session_state.accessibility.get('high_contrast', False) else 'black')
            
            # Appliquer le fond sombre en mode contraste élevé
            if st.session_state.accessibility.get('high_contrast', False):
                plt.gca().set_facecolor('black')
                plt.gcf().set_facecolor('black')
            
            st.pyplot(plt)
        else:
            st.warning("⚠️ Aucun mot-clé disponible pour générer le nuage de mots.")
    
except FileNotFoundError:
    st.error(f"❌ Fichier {KEYWORD_FREQ_PATH} non trouvé.")

# Données visuelles non structurées
st.subheader("Données Visuelles Non Structurées")
st.write("**Exemple d'image par catégorie :**")

# Debugging: Display category and image availability
st.write("**Disponibilité des images par catégorie :**")
for category in df['main_category'].unique()[:3]:
    count = len(df[(df['main_category'] == category) & df['image_exists'] & (df['image_pixels'] > 0)])
    st.write(f"- {category}: {count} images valides (image_pixels > 0)")

for category in df['main_category'].unique()[:3]:
    st.write(f"**Catégorie : {category}**")
    filtered_df = df[(df['main_category'] == category) & df['image_exists'] & (df['image_pixels'] > 0)]['image']
    if not filtered_df.empty:
        sample_paths = filtered_df.sample(n=1, random_state=42)
        for path in sample_paths:
            full_path = f"Images/{path}"
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path)
                    st.image(img, caption=f"Exemple pour {category}", width=200)
                    # Texte alternatif pour les images
                    st.caption(f"Image d'exemple pour la catégorie {category}")
                except Exception as e:
                    st.write(f"⚠️ Impossible de charger l'image pour {category}: {str(e)}")
            else:
                st.write(f"⚠️ Chemin d'image non valide pour {category}: {full_path}")
    else:
        st.write(f"Aucune image valide disponible pour la catégorie {category} (aucune image avec image_pixels > 0).")

# Statistiques sur les images
st.write("**Statistiques sur les images :**")
valid_image_df = df[df['image_pixels'] > 0][['image_pixels', 'aspect_ratio']]
if valid_image_df.empty:
    st.warning("⚠️ Aucune image valide (image_pixels > 0) disponible pour les statistiques.")
else:
    st.write(f"**Nombre d'images valides :** {len(valid_image_df)}")
    st.dataframe(valid_image_df.describe())

# Scatter plot accessible
if not valid_image_df.empty and 'aspect_ratio' in valid_image_df.columns and 'image_pixels' in valid_image_df.columns:
    fig5 = px.scatter(valid_image_df, x='aspect_ratio', y='image_pixels', 
                      color=df.loc[valid_image_df.index, 'main_category'],
                      title="Ratio Hauteur/Largeur vs Nombre de Pixels (Images Valides)",
                      color_discrete_sequence=PLOTLY_COLORS,
                      labels={'aspect_ratio': 'Ratio Hauteur/Largeur', 'image_pixels': 'Nombre de Pixels'})
    fig5.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(size=12 if not st.session_state.accessibility.get('large_text', False) else 16, color=text_color),
        legend_title="Catégories principales",
        legend=dict(font=dict(color=text_color)),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14 if not st.session_state.accessibility.get('large_text', False) else 16,
            font_family="Arial, sans-serif",
            font_color="black",
            bordercolor="black"
        )
    )
    fig5.update_traces(
        marker=dict(size=8, opacity=0.7),
        selector=dict(mode='markers')
    )
    fig5.update_xaxes(
        tickfont=dict(color=text_color, size=12 if not st.session_state.accessibility.get('large_text', False) else 16)
    )
    fig5.update_yaxes(
        tickfont=dict(color=text_color, size=12 if not st.session_state.accessibility.get('large_text', False) else 16)
    )
    st.plotly_chart(fig5, use_container_width=True, aria_label="Nuage de points du ratio hauteur/largeur vs nombre de pixels")
else:
    st.warning("⚠️ Données insuffisantes pour afficher le nuage de points (aucune image valide avec aspect_ratio ou image_pixels).")

# Debugging: Display invalid images
st.write("**Images invalides (image_pixels = 0 ou manquant) :**")
invalid_images = df[df['image_pixels'] <= 0][['image', 'main_category', 'image_pixels']]
if not invalid_images.empty:
    st.dataframe(invalid_images)
else:
    st.write("✅ Toutes les images ont des valeurs valides pour image_pixels.")

# Afficher les options d'accessibilité dans la sidebar
render_accessibility_sidebar()

# Appliquer les styles d'accessibilité
apply_accessibility_styles()