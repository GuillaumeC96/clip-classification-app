#!/usr/bin/env python3
"""
Test d'intégration complète : Backend Azure ML + Frontend Streamlit
"""

import requests
import json
from PIL import Image
import io
import base64

def test_backend_azure_ml():
    """Test du backend Azure ML"""
    print("🔍 Test du Backend Azure ML...")
    
    # Test 1: Health check
    try:
        response = requests.get("https://clip-onnx-interpretability.azurewebsites.net/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check: OK")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Health check: ERROR - {e}")
        return False
    
    # Test 2: Prédiction simple
    try:
        data = {
            "text": "montre analogique pour homme",
            "product_keywords": "watch, analog, men, stainless, steel"
        }
        response = requests.post(
            "https://clip-onnx-interpretability.azurewebsites.net/score",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prédiction: OK")
            print(f"   Catégorie: {result.get('predicted_category')}")
            print(f"   Confiance: {result.get('confidence')}")
            print(f"   Source: {result.get('source')}")
            
            # Vérifier l'interprétabilité
            if 'attention_result' in result:
                print("✅ Interprétabilité: OK")
                attention = result['attention_result']
                if 'keyword_similarities' in attention:
                    print(f"   Mots-clés: {list(attention['keyword_similarities'].keys())}")
                if 'heatmap' in attention:
                    print(f"   Heatmap: {len(attention['heatmap'])}x{len(attention['heatmap'][0])} pixels")
            else:
                print("⚠️ Interprétabilité: Non disponible")
            
            return True
        else:
            print(f"❌ Prédiction: FAILED ({response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prédiction: ERROR - {e}")
        return False

def test_frontend_streamlit():
    """Test du frontend Streamlit"""
    print("\n🔍 Test du Frontend Streamlit...")
    
    try:
        response = requests.get("http://localhost:8506", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend Streamlit: OK")
            if "Streamlit" in response.text:
                print("   Interface Streamlit détectée")
            return True
        else:
            print(f"❌ Frontend Streamlit: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Frontend Streamlit: ERROR - {e}")
        return False

def test_client_azure():
    """Test du client Azure ML"""
    print("\n🔍 Test du Client Azure ML...")
    
    try:
        # Importer le client
        import sys
        sys.path.append('.')
        from azure_client import get_azure_client
        
        # Créer le client
        client = get_azure_client(show_warning=False)
        
        print(f"✅ Client créé: OK")
        print(f"   Endpoint: {client.endpoint_url}")
        print(f"   Source: {client.config_source}")
        print(f"   ONNX: {client.is_onnx}")
        print(f"   Simulé: {client.use_simulated}")
        
        # Test de prédiction
        # Créer une image de test
        test_image = Image.new('RGB', (224, 224), color='white')
        
        result = client.predict_category(
            test_image, 
            "montre analogique pour homme", 
            "watch, analog, men, stainless, steel"
        )
        
        if result['success']:
            print("✅ Prédiction client: OK")
            print(f"   Catégorie: {result['predicted_category']}")
            print(f"   Confiance: {result['confidence']}")
            print(f"   Source: {result['source']}")
            return True
        else:
            print(f"❌ Prédiction client: FAILED")
            print(f"   Erreur: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Client Azure ML: ERROR - {e}")
        return False

def main():
    """Test d'intégration complet"""
    print("🚀 TEST D'INTÉGRATION COMPLET")
    print("=" * 50)
    
    # Tests
    backend_ok = test_backend_azure_ml()
    frontend_ok = test_frontend_streamlit()
    client_ok = test_client_azure()
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    print(f"Backend Azure ML: {'✅ OK' if backend_ok else '❌ FAILED'}")
    print(f"Frontend Streamlit: {'✅ OK' if frontend_ok else '❌ FAILED'}")
    print(f"Client Azure ML: {'✅ OK' if client_ok else '❌ FAILED'}")
    
    if backend_ok and frontend_ok and client_ok:
        print("\n🎉 SYSTÈME COMPLET FONCTIONNEL !")
        print("✅ Backend Azure ML opérationnel")
        print("✅ Frontend Streamlit opérationnel") 
        print("✅ Client Azure ML opérationnel")
        print("✅ Intégration complète réussie")
    else:
        print("\n🚨 PROBLÈMES DÉTECTÉS")
        if not backend_ok:
            print("❌ Backend Azure ML non fonctionnel")
        if not frontend_ok:
            print("❌ Frontend Streamlit non fonctionnel")
        if not client_ok:
            print("❌ Client Azure ML non fonctionnel")

if __name__ == "__main__":
    main()
