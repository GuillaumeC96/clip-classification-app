"""
Backend Azure ML pour l'inférence CLIP avec vraies heatmaps d'attention
Implémentation basée sur le notebook Cassez_Guillaume_3_notebook_POC_082025.ipynb
"""

import os
import json
import base64
import io
import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.interpolate import griddata
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, request, jsonify
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Variables globales pour le modèle CLIP
model = None
processor = None
device = None

def load_clip_model():
    """Charger le modèle CLIP fine-tuné"""
    global model, processor, device
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Device: {device}")
        
        # Charger le modèle CLIP fine-tuné (ou le modèle de base si pas de fine-tuning)
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        logger.info("✅ Modèle CLIP chargé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle CLIP: {str(e)}")
        return False

def preprocess_text(text_description, product_keywords=None):
    """Préprocesser le texte comme dans le notebook"""
    # Nettoyer et extraire les mots-clés
    import re
    
    # Combiner description et mots-clés
    full_text = f"{text_description}"
    if product_keywords:
        full_text += f" {product_keywords}"
    
    # Nettoyer le texte
    full_text = re.sub(r'[^\w\s]', ' ', full_text.lower())
    words = full_text.split()
    
    # Filtrer les mots vides et courts
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # Prendre les mots les plus fréquents
    from collections import Counter
    keyword_counts = Counter(keywords)
    top_keywords = [word for word, count in keyword_counts.most_common(10)]
    
    return top_keywords

def generate_clip_attention_heatmap(image, text_description, product_keywords=None):
    """
    Générer une vraie heatmap d'attention CLIP
    Logique identique au notebook Cassez_Guillaume_3_notebook_POC_082025.ipynb
    """
    try:
        # Convertir l'image en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Améliorer le contraste comme dans le notebook
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Créer une version noir et blanc pour l'affichage
        img_bw = image.convert('L')
        img_width, img_height = image.size
        
        # Préprocesser le texte
        keywords = preprocess_text(text_description, product_keywords)
        
        if not keywords:
            logger.warning("⚠️ Aucun mot-clé extrait du texte")
            return None
        
        logger.info(f"🔍 Mots-clés extraits: {keywords}")
        
        # Paramètres de la heatmap (identiques au notebook)
        resolution = 50  # Densité d'échantillonnage
        patch_size = 128
        step = patch_size
        
        # Créer la grille de positions
        x = np.linspace(0, img_width, resolution, dtype=int)
        y = np.linspace(0, img_height, resolution, dtype=int)
        xx, yy = np.meshgrid(x, y)
        
        # Extraire les patches et calculer les features
        positions = []
        patch_features = []
        batch_size = 10
        
        for i in range(0, resolution * resolution, batch_size):
            batch_patches = []
            batch_positions = []
            
            for j in range(i, min(i + batch_size, resolution * resolution)):
                x_idx = j // resolution
                y_idx = j % resolution
                x_pos = xx[x_idx, y_idx]
                y_pos = yy[x_idx, y_idx]
                
                # Extraire le patch
                patch = image.crop((
                    max(0, x_pos - patch_size//2), 
                    max(0, y_pos - patch_size//2),
                    min(img_width, x_pos + patch_size//2), 
                    min(img_height, y_pos + patch_size//2)
                ))
                
                if patch.size[0] > 0 and patch.size[1] > 0:
                    # Redimensionner le patch comme dans le notebook
                    patch = patch.resize((224, 224), Image.LANCZOS)
                    batch_patches.append(patch)
                    batch_positions.append((x_pos, y_pos))
            
            if batch_patches:
                with torch.no_grad():
                    # Traiter le batch de patches
                    inputs = processor(images=batch_patches, return_tensors="pt").pixel_values.to(device).float()
                    
                    if inputs.shape[1] != 3:  # Vérifier 3 canaux RGB
                        logger.warning("⚠️ Batch non-RGB détecté, saut du batch")
                        continue
                    
                    # Calculer les features d'image
                    features = model.get_image_features(pixel_values=inputs)
                    patch_features.append(features)
                
                positions.extend(batch_positions)
                torch.cuda.empty_cache()
        
        if not patch_features:
            raise ValueError("Aucun patch valide extrait pour la heatmap")
        
        # Concaténer les features et normaliser
        patch_features = torch.cat(patch_features)
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
        
        # Calculer les features de texte
        with torch.no_grad():
            text_inputs = processor(text=keywords, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculer les scores d'attention (logique identique au notebook)
            attention_scores = (patch_features @ text_features.T).cpu().numpy()
        
        # Interpolation pour créer la heatmap lisse
        points = np.array(positions)
        grid_x, grid_y = np.mgrid[0:img_width:complex(0, img_width), 0:img_height:complex(0, img_height)]
        
        # Utiliser la moyenne des scores d'attention comme dans le notebook
        smooth_heatmap = griddata(
            points, 
            attention_scores.mean(axis=1), 
            (grid_x, grid_y), 
            method='cubic', 
            fill_value=0
        )
        
        # Normaliser la heatmap
        smooth_heatmap = (smooth_heatmap - smooth_heatmap.min()) / (smooth_heatmap.max() - smooth_heatmap.min())
        
        # Calculer les similarités par mot-clé
        keyword_similarities = {}
        for i, keyword in enumerate(keywords):
            keyword_similarities[keyword] = float(attention_scores.mean(axis=0)[i])
        
        # Trier par similarité
        sorted_keywords = sorted(keyword_similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Top keywords (convertir en format JSON-serializable)
        top_keywords = [(kw, float(score)) for kw, score in sorted_keywords[:3]]
        
        logger.info(f"✅ Heatmap générée: {smooth_heatmap.shape}")
        logger.info(f"🔝 Top keywords: {top_keywords}")
        
        return {
            'heatmap': smooth_heatmap.tolist(),
            'keyword_similarities': keyword_similarities,
            'keywords': keywords,
            'positions': [[int(x), int(y)] for x, y in positions],  # Convertir en int
            'attention_scores': attention_scores.tolist(),
            'top_keywords': top_keywords
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération de heatmap: {str(e)}")
        return None

@app.route('/score', methods=['POST'])
def score():
    """Endpoint principal pour les prédictions et heatmaps"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée fournie'}), 400
        
        # Vérifier si c'est une requête de heatmap
        action = data.get('action', 'predict')
        
        if action == 'heatmap':
            # Générer une heatmap d'attention
            if 'image' not in data:
                return jsonify({'error': 'Image requise pour la heatmap'}), 400
            
            # Décoder l'image
            try:
                image_b64 = data['image']
                image_data = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_data))
            except Exception as e:
                return jsonify({'error': f'Erreur de décodage image: {str(e)}'}), 400
            
            # Générer la heatmap
            text_description = data.get('text_description', '')
            product_keywords = data.get('product_keywords', '')
            
            heatmap_result = generate_clip_attention_heatmap(image, text_description, product_keywords)
            
            if heatmap_result:
                return jsonify({
                    'heatmap': heatmap_result['heatmap'],
                    'keyword_similarities': heatmap_result['keyword_similarities'],
                    'keywords': heatmap_result['keywords'],
                    'positions': heatmap_result['positions'],
                    'attention_scores': heatmap_result['attention_scores'],
                    'top_keywords': heatmap_result['top_keywords'],
                    'message': 'Heatmap CLIP générée avec succès',
                    'source': 'azure_clip_real'
                })
            else:
                return jsonify({'error': 'Impossible de générer la heatmap'}), 500
        
        else:
            # Prédiction normale (simulation pour l'instant)
            text = data.get('text', '')
            product_keywords = data.get('product_keywords', '')
            
            # Simulation simple de prédiction
            categories = ['Watches', 'Shoes', 'Bags', 'Clothing', 'Electronics']
            predicted_category = np.random.choice(categories)
            confidence = np.random.uniform(0.7, 0.95)
            
            return jsonify({
                'predicted_category': predicted_category,
                'confidence': confidence,
                'message': 'Prédiction simulée - Backend CLIP en cours de développement',
                'source': 'azure_clip_backend'
            })
    
    except Exception as e:
        logger.error(f"❌ Erreur dans l'endpoint /score: {str(e)}")
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown'
    })

if __name__ == '__main__':
    # Charger le modèle au démarrage
    if load_clip_model():
        logger.info("🚀 Démarrage du serveur Azure ML Backend")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("❌ Impossible de démarrer le serveur - modèle non chargé")
