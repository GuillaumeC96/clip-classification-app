"""
Client Azure ML pour l'inférence du modèle CLIP
"""

import os
import json
import base64
import requests
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
        """Prédiction via l'API Azure ML avec interprétabilité (LOGIQUE IDENTIQUE AU NOTEBOOK)"""
        try:
            # Encoder l'image
            image_base64 = self.encode_image_to_base64(image)
            
            # Préparer les données avec l'image pour l'interprétabilité
            # Format compatible avec le service Azure ML actuel
            data = {
                "image": image_base64,  # Ajouter l'image pour l'interprétabilité
                "text": text_description,  # Utiliser 'text' au lieu de 'text_description'
                "product_keywords": product_keywords  # Mots-clés du CSV
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
                        'category_scores': result.get('category_scores', {}),
                        'source': result.get('source', 'azure_ml'),
                        'attention_result': result.get('attention_result')  # Ajouter les résultats d'interprétabilité
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
                'category_scores': scores,
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
