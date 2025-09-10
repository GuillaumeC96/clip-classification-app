"""
Client Azure ML pour l'inférence du modèle CLIP
"""

import os
import json
import base64
import requests
import re
import streamlit as st
from PIL import Image
import io
from typing import Dict, Any, Optional
from dotenv import load_dotenv
# from streamlit_secrets_config import get_azure_config  # Module supprimé

# Charger les variables d'environnement depuis .env_azure_production (pour le développement)
# Désactivé pour Streamlit Cloud - configuration par défaut hardcodée
# load_dotenv('.env_azure_production')

class AzureMLClient:
    """Client pour interagir avec l'API Azure ML"""
    
    def __init__(self, show_warning=True):
        # Configuration par défaut - sera remplacée par get_azure_config()
        self.endpoint_url = None
        self.api_key = None
        self.is_simulated = False
        self.use_simulated = False
        self.config_source = 'none'
        
        # Vérifier si c'est un endpoint ONNX (Azure ML ou avec 'onnx' dans l'URL)
        self.is_onnx = (self.endpoint_url and 
                       ('onnx' in self.endpoint_url.lower() or 
                        'inference.ml.azure.com' in self.endpoint_url.lower() or
                        'azureml' in self.endpoint_url.lower()))
        
        # Configuration par défaut - Endpoint Azure ML cloud
        config = {
            'endpoint_url': "https://clip-onnx-interpretability.azurewebsites.net/score",
            'api_key': "dummy_key",
            'source': 'default_azure_ml'
        }
        
        # Vérifier s'il y a des secrets Streamlit configurés (IGNORER pour utiliser le bon endpoint)
        # try:
        #     import streamlit as st
        #     if hasattr(st, 'secrets') and hasattr(st.secrets, 'get'):
        #         azure_endpoint = st.secrets.get('AZURE_ML_ENDPOINT_URL')
        #         azure_key = st.secrets.get('AZURE_ML_API_KEY')
        #         if azure_endpoint and azure_key:
        #             config = {
        #                 'endpoint_url': azure_endpoint,
        #                 'api_key': azure_key,
        #                 'source': 'streamlit_secrets'
        #             }
        # except Exception:
        #     # Ignorer les erreurs de secrets
        #     pass
        # Toujours utiliser la configuration trouvée
        self.endpoint_url = config['endpoint_url']
        self.api_key = config['api_key']
        self.config_source = config['source']
        
        # Recalculer les propriétés
        self.is_onnx = (self.endpoint_url and 
                       ('onnx' in self.endpoint_url.lower() or 
                        'inference.ml.azure.com' in self.endpoint_url.lower() or
                        'azureml' in self.endpoint_url.lower()))
        self.is_simulated = self.endpoint_url and 'simulated' in self.endpoint_url.lower()
        
        # Gérer le mode simulé - par défaut activé pour plan gratuit
        if self.is_simulated or not self.endpoint_url:
            self.use_simulated = True
        else:
            # Pour plan gratuit, utiliser le mode simulé par défaut
            self.use_simulated = os.getenv('USE_SIMULATED_MODEL', 'true').lower() == 'true'
        
        # Afficher le statut de la configuration
        if show_warning:
            if self.config_source == 'streamlit_secrets':
                # Vérifier si ce sont des secrets par défaut ou de vrais secrets
                is_default_config = (self.endpoint_url == "https://your-endpoint.westeurope.inference.ml.azure.com/score" and 
                                   self.api_key == "your_api_key_here")
                
                if is_default_config:
                    # Ce sont des secrets par défaut, traiter comme azure_onnx_default
                    is_cloud = os.getenv('STREAMLIT_SERVER_ENVIRONMENT') == 'cloud'
                    
                    if is_cloud:
                        # Sur Streamlit Cloud, afficher un message informatif sans avertissement
                        st.info("ℹ️ Configuration Azure ML ONNX par défaut")
                        st.info("💡 Pour utiliser votre endpoint Azure ML, configurez les secrets dans Streamlit Cloud")
                        st.info("📋 Voir la section 'Configuration' ci-dessous pour plus d'informations")
                    else:
                        # En développement, afficher le message complet
                        st.success("🚀 Configuration Azure ML ONNX par défaut activée")
                        st.info("✅ Modèles ONNX optimisés pour des performances maximales")
                        st.info("💡 Configuration par défaut - Prêt pour l'inférence ONNX")
                        if self.is_onnx:
                            st.success("🎯 Endpoint ONNX détecté - Performances optimisées")
                else:
                    # Ce sont de vrais secrets Streamlit Cloud
                    st.success("🚀 Configuration Azure ML chargée depuis Streamlit Cloud")
                    if self.is_simulated:
                        st.info("🎭 Mode simulé activé - Prédictions intelligentes avec mots-clés")
                    elif self.is_onnx:
                        st.success("🚀 Client Azure ML ONNX configuré (performances maximales)")
                    else:
                        st.info("✅ Client Azure ML configuré")
            elif self.config_source == 'env_vars':
                st.info("✅ Configuration Azure ML chargée depuis les variables d'environnement")
                if self.is_simulated:
                    st.info("🎭 Mode simulé activé - Prédictions intelligentes avec mots-clés")
                elif self.is_onnx:
                    st.success("🚀 Client Azure ML ONNX configuré (performances maximales)")
                else:
                    st.info("✅ Client Azure ML configuré")
            elif self.config_source == 'default_simulated':
                # Mode démonstration par défaut - optimisé pour plan gratuit
                st.success("🎭 Mode démonstration activé (Plan Azure gratuit)")
                st.info("✅ Prédictions intelligentes avec analyse de mots-clés")
                st.info("💡 L'application fonctionne avec des prédictions simulées")
                st.info("🆓 Optimisé pour le plan Azure gratuit - Évite les limitations de ressources")
                st.info("💡 Pour utiliser Azure ML, passez à un plan payant avec plus de ressources")
            else:
                # Mode démonstration amélioré - pas d'avertissement alarmant
                st.success("🎭 Mode démonstration activé")
                st.info("✅ Prédictions intelligentes avec analyse de mots-clés")
                st.info("💡 L'application fonctionne avec des prédictions simulées")
                st.info("🔧 Configuration par défaut - Évite les problèmes de timeout")
                st.info("💡 Pour utiliser Azure ML, configurez un endpoint valide dans les secrets")
                self.use_simulated = True
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convertir une image PIL en base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _preprocess_image_like_notebook(self, image: Image.Image) -> Image.Image:
        """Prétraitement de l'image identique au notebook (extract_image_features)"""
        try:
            # Convertir en RGB si nécessaire (comme dans le notebook)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionner si nécessaire (comme dans le notebook)
            max_size = 128  # Même valeur que dans le notebook
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)  # Même méthode que le notebook
            
            return image
        except Exception as e:
            print(f"⚠️ Erreur prétraitement image: {str(e)}")
            return image
    
    def _preprocess_text_like_notebook(self, text_description: str, product_keywords: str = None) -> str:
        """Prétraitement du texte identique au notebook (clean_text + extract_keywords)"""
        try:
            # Combiner description et mots-clés
            combined_text = f"{text_description}"
            if product_keywords:
                combined_text += f" {product_keywords}"
            
            # Appliquer le nettoyage identique au notebook
            cleaned_text = self._clean_text_like_notebook(combined_text)
            
            # Extraire les mots-clés comme dans le notebook
            keywords = self._extract_keywords_like_notebook(cleaned_text)
            
            # Retourner les mots-clés séparés par des virgules (comme dans le notebook)
            return ", ".join(keywords) if keywords else "no_keywords_found"
            
        except Exception as e:
            print(f"⚠️ Erreur prétraitement texte: {str(e)}")
            return text_description
    
    def _clean_text_like_notebook(self, text: str) -> str:
        """Nettoyage du texte identique au notebook (clean_text function)"""
        if not text:
            return ""
        
        # Règles de nettoyage identiques au notebook
        all_patterns = [
            # Ponctuation spécifique
            (r'\(', ' ( '),
            (r'\)', ' ) '),
            (r'\.', ' . '),
            (r'\!', ' ! '),
            (r'\?', ' ? '),
            (r'\:', ' : '),
            (r'\,', ', '),
            # Motifs spécifiques du domaine
            (r'\b(\d+)\s*[-~to]?\s*(\d+)\s*(m|mth|mths|month|months?)\b', 'month'),
            (r'\bnewborn\s*[-~to]?\s*(\d+)\s*(m|mth|months?)\b', 'month'),
            (r'\b(nb|newborn|baby|bb|bby|babie|babies)\b', 'baby'),
            (r'\b(diaper|diapr|nappy)\b', 'diaper'),
            (r'\b(stroller|pram|buggy)\b', 'stroller'),
            (r'\b(bpa\s*free|non\s*bpa)\b', 'bisphenol a free'),
            (r'\b(\d+)\s*(oz|ounce)\b', 'ounce'),
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
            (r'\bfridge\b', 'refrigerator'),
            (r'\bwashing\s*machine\b', 'clothes washer'),
            (r'\bdishwasher\b', 'dish washing machine'),
            (r'\boven\b', 'cooking oven'),
            (r'\bmicrowave\b', 'microwave oven'),
            (r'\bhoover\b', 'vacuum cleaner'),
            (r'\btumble\s*dryer\b', 'clothes dryer'),
            (r'\b(a\+\++)\b', 'energy efficiency class'),
            (r'\b(\d+)\s*btu\b', 'british thermal unit'),
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
            (r'\bSPF\b', 'sun protection factor'),
            (r'\bUV\b', 'ultraviolet'),
            (r'\bBB\s*cream\b', 'blemish balm cream'),
            (r'\bCC\s*cream\b', 'color correcting cream'),
            (r'\bHA\b', 'hyaluronic acid'),
            (r'\bAHA\b', 'alpha hydroxy acid'),
            (r'\bBHA\b', 'beta hydroxy acid'),
            (r'\bPHA\b', 'polyhydroxy acid'),
            (r'\bNMF\b', 'natural moisturizing factor'),
            (r'\bEGF\b', 'epidermal growth factor'),
            (r'\bVit\s*C\b', 'vitamin c'),
            (r'\bVit\s*E\b', 'vitamin e'),
            (r'\bVit\s*B3\b', 'niacinamide vitamin b3'),
            (r'\bVit\s*B5\b', 'panthenol vitamin b5'),
            (r'\bSOD\b', 'superoxide dismutase'),
            (r'\bQ10\b', 'coenzyme q10'),
            (r'\bFoam\s*cl\b', 'foam cleanser'),
            (r'\bMic\s*H2O\b', 'micellar water'),
            (r'\bToner\b', 'skin toner'),
            (r'\bEssence\b', 'skin essence'),
            (r'\bAmpoule\b', 'concentrated serum'),
            (r'\bCF\b', 'cruelty free'),
            (r'\bPF\b', 'paraben free'),
            (r'\bSF\b', 'sulfate free'),
            (r'\bGF\b', 'gluten free'),
            (r'\bHF\b', 'hypoallergenic formula'),
            (r'\bNT\b', 'non-comedogenic tested'),
            (r'\bAM\b', 'morning'),
            (r'\bPM\b', 'night'),
            (r'\bBID\b', 'twice daily'),
            (r'\bQD\b', 'once daily'),
            (r'\bAIR\b', 'airless pump bottle'),
            (r'\bD-C\b', 'dropper container'),
            (r'\bT-C\b', 'tube container'),
            (r'\bPDO\b', 'polydioxanone'),
            (r'\bPCL\b', 'polycaprolactone'),
            (r'\bPLLA\b', 'poly-l-lactic acid'),
            (r'\bHIFU\b', 'high-intensity focused ultrasound'),
            (r'\b(\d+)\s*fl\s*oz\b', 'fluid ounce'),
            (r'\bpH\s*bal\b', 'ph balanced'),
            (r'\b(\d+)\s*(gb|tb|mb|go|to|mo)\b', 'byte'),
            (r'\boctet\b', 'byte'),
            (r'\b(\d+)\s*y\b', 'year'),
            (r'\b(\d+)\s*mth\b', 'month'),
            (r'\b(\d+)\s*d\b', 'day'),
            (r'\b(\d+)\s*h\b', 'hour'),
            (r'\b(\d+)\s*min\b', 'minute'),
            (r'\b(\d+)\s*rpm\b', 'revolution per minute'),
            (r'\b(\d+)\s*(mw|cw|kw)\b', 'watt'),
            (r'\b(\d+)\s*(ma|ca|ka)\b', 'ampere'),
            (r'\b(\d+)\s*(mv|cv|kv)\b', 'volt'),
            (r'\b(\d+)\s*(mm|cm|m|km)\b', 'meter'),
            (r'\binch\b', 'meter'),
            (r'\b(\d+)\s*(ml|cl|dl|l|oz|gal)\b', 'liter'),
            (r'\b(gallon|ounce)\b', 'liter'),
            (r'\b(\d+)\s*(mg|cg|dg|g|kg|lb)\b', 'gram'),
            (r'\bpound\b', 'gram'),
            (r'\b(\d+)\s*(°c|°f)\b', 'celsius'),
            (r'\bfahrenheit\b', 'celsius'),
            (r'\bflipkart\.com\b', ''),
            (r'\bapprox\.?\b', 'approximately'),
            (r'\bw/o\b', 'without'),
            (r'\bw/\b', 'with'),
            (r'\bant-\b', 'anti'),
            (r'\byes\b', ''),
            (r'\bno\b', ''),
            (r'\bna\b', ''),
            (r'\brs\.?\b', ''),
            # Normaliser les espaces
            (r'\s+', ' '),
        ]
        
        # Appliquer les patterns deux fois comme dans le notebook
        for _ in range(2):
            for pattern, replacement in all_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_keywords_like_notebook(self, text: str, top_n: int = 15) -> list:
        """Extraction de mots-clés identique au notebook (extract_keywords function)"""
        if not text:
            return []
        
        # Mots vides à ignorer (comme dans le notebook)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Nettoyage et extraction des mots
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = []
        
        for word in words:
            # Filtrer comme dans le notebook
            if (len(word) >= 2 and 
                word not in stop_words and 
                not re.match(r'.*[@*/±&%#].*', word)):
                keywords.append(word)
        
        # Compter et retourner les top mots-clés
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(top_n)]
    
    def predict_category(self, image: Image.Image, text_description: str, product_keywords: str = None) -> Dict[str, Any]:
        """
        Prédire la catégorie d'un produit (LOGIQUE IDENTIQUE AU NOTEBOOK)
        
        Args:
            image: Image PIL du produit
            text_description: Description textuelle du produit
            product_keywords: Mots-clés du produit depuis le CSV
            
        Returns:
            Dict contenant les résultats de prédiction
        """
        if self.use_simulated:
            return self._predict_simulated(image, text_description)
        else:
            return self._predict_azure(image, text_description, product_keywords)
    
    def _predict_azure(self, image: Image.Image, text_description: str, product_keywords: str = None) -> Dict[str, Any]:
        """Prédiction via l'API Azure ML avec prétraitement identique au notebook"""
        try:
            # PRÉTRAITEMENT IDENTIQUE AU NOTEBOOK
            
            # 1. Prétraitement de l'image (comme dans extract_image_features)
            processed_image = self._preprocess_image_like_notebook(image)
            image_base64 = self.encode_image_to_base64(processed_image)
            
            # 2. Prétraitement du texte (comme dans clean_text + extract_keywords)
            processed_text = self._preprocess_text_like_notebook(text_description, product_keywords)
            
            # Préparer les données avec le prétraitement identique au notebook
            data = {
                "image": image_base64,
                "text": processed_text,  # Texte prétraité comme dans le notebook
                "product_keywords": product_keywords
            }
            
            # Headers pour l'authentification
            headers = {
                'Content-Type': 'application/json'
            }
            
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            # Appel à l'API
            response = requests.post(
                self.endpoint_url,
                data=json.dumps(data),
                headers=headers,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                # Vérifier si la réponse contient les données attendues
                if 'predicted_category' in result or 'attention_result' in result:
                    return {
                        'success': True,
                        'predicted_category': result.get('predicted_category'),
                        'confidence': result.get('confidence', 0),
                        'source': result.get('source', 'azure_ml')
                    }
                else:
                    return {
                        'success': False,
                        'error': result.get('error', 'Erreur inconnue de l\'API'),
                        'source': 'azure_ml'
                    }
            elif response.status_code == 503:
                # Service indisponible - NE PAS basculer automatiquement
                return {
                    'success': False,
                    'error': f'Service Azure ML indisponible (503): {response.text}',
                    'source': 'azure_ml'
                }
            else:
                return {
                    'success': False,
                    'error': f'Erreur HTTP {response.status_code}: {response.text}',
                    'source': 'azure_ml'
                }
                
        except requests.exceptions.Timeout:
            # Timeout - NE PAS basculer automatiquement
            return {
                'success': False,
                'error': f'Timeout Azure ML: {str(e)}',
                'source': 'azure_ml'
            }
        except requests.exceptions.RequestException as e:
            # Erreur de connexion - NE PAS basculer automatiquement
            return {
                'success': False,
                'error': f'Erreur de connexion Azure ML: {str(e)}',
                'source': 'azure_ml'
            }
        except Exception as e:
            # Erreur inattendue - NE PAS basculer automatiquement
            return {
                'success': False,
                'error': f'Erreur inattendue Azure ML: {str(e)}',
                'source': 'azure_ml'
            }
    
    
    def _predict_simulated(self, image: Image.Image, text_description: str) -> Dict[str, Any]:
        """Prédiction simulée intelligente (optimisée pour plan gratuit)"""
        try:
            # Simulation d'une prédiction basée sur des règles intelligentes
            combined_text = text_description.lower()
            
            # Catégories disponibles
            categories = [
                'Baby Care', 'Beauty and Personal Care', 'Computers',
                'Home Decor & Festive Needs', 'Home Furnishing',
                'Kitchen & Dining', 'Watches'
            ]
            
            # Règles améliorées basées sur les mots-clés (plus intelligentes)
            category_keywords = {
                'Baby Care': ['baby', 'enfant', 'bébé', 'nourrisson', 'couche', 'jouet', 'kids', 'child', 'toddler', 'infant', 'stroller', 'pram'],
                'Beauty and Personal Care': ['beauté', 'cosmétique', 'soin', 'shampooing', 'crème', 'maquillage', 'beauty', 'care', 'skin', 'hair', 'makeup', 'lotion', 'serum', 'moisturizer'],
                'Computers': ['ordinateur', 'laptop', 'pc', 'computer', 'écran', 'clavier', 'desktop', 'monitor', 'keyboard', 'mouse', 'gaming', 'graphics', 'processor'],
                'Home Decor & Festive Needs': ['déco', 'décoration', 'fête', 'festif', 'ornement', 'decor', 'decoration', 'ornament', 'festive', 'wall', 'art', 'frame'],
                'Home Furnishing': ['meuble', 'furniture', 'canapé', 'table', 'chaise', 'lit', 'sofa', 'chair', 'bed', 'table', 'furniture', 'couch', 'dining'],
                'Kitchen & Dining': ['cuisine', 'kitchen', 'vaisselle', 'casserole', 'four', 'réfrigérateur', 'cookware', 'dining', 'plate', 'bowl', 'utensil', 'appliance'],
                'Watches': ['montre', 'watch', 'horloge', 'chronomètre', 'bracelet', 'sapphero', 'watches', 'timepiece', 'clock', 'stainless', 'steel', 'quartz', 'water', 'resistant', 'analog', 'digital', 'wrist']
            }
            
            # Calculer les scores avec pondération intelligente
            scores = {}
            for category, keywords in category_keywords.items():
                # Score basé sur le nombre de mots-clés trouvés
                matches = sum(1 for keyword in keywords if keyword in combined_text)
                # Score normalisé par le nombre de mots-clés
                base_score = matches / len(keywords)
                
                # Bonus pour les mots-clés très spécifiques
                specific_bonus = 0
                if category == 'Watches' and any(word in combined_text for word in ['analog', 'digital', 'stainless', 'steel', 'quartz', 'water', 'resistant', 'wrist']):
                    specific_bonus = 0.3
                elif category == 'Computers' and any(word in combined_text for word in ['laptop', 'desktop', 'monitor', 'gaming', 'graphics']):
                    specific_bonus = 0.25
                elif category == 'Beauty and Personal Care' and any(word in combined_text for word in ['beauty', 'care', 'skin', 'hair', 'makeup', 'serum']):
                    specific_bonus = 0.2
                elif category == 'Kitchen & Dining' and any(word in combined_text for word in ['kitchen', 'cookware', 'appliance', 'utensil']):
                    specific_bonus = 0.2
                
                scores[category] = min(1.0, base_score + specific_bonus)
            
            # Prédiction intelligente
            if max(scores.values()) > 0:
                predicted_category = max(scores, key=scores.get)
                confidence = max(scores.values())
                # Améliorer la confiance si plusieurs mots-clés correspondent
                if confidence > 0.2:
                    confidence = min(0.92, confidence + 0.15)
                elif confidence > 0.1:
                    confidence = min(0.85, confidence + 0.1)
            else:
                predicted_category = 'Home Furnishing'  # Catégorie par défaut
                confidence = 0.15
            
            return {
                'success': True,
                'predicted_category': predicted_category,
                'confidence': confidence,
                'source': 'demo_optimized'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur lors de la prédiction de démonstration: {str(e)}',
                'source': 'demo'
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Vérifier le statut du service Azure ML"""
        if self.use_simulated:
            return {
                'status': 'simulated',
                'message': 'Utilisation du modèle simulé'
            }
        
        try:
            # Test simple de connectivité
            response = requests.get(
                self.endpoint_url.replace('/score', '/health'),
                timeout=5
            )
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'message': f'Service Azure ML - Status: {response.status_code}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Impossible de contacter le service: {str(e)}'
            }

# Instance globale du client
@st.cache_resource
def get_azure_client(show_warning=True):
    """Obtenir l'instance du client Azure ML"""
    return AzureMLClient(show_warning=show_warning)
