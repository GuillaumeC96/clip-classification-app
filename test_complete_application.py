"""
Test complet de l'application avec le nouveau backend CLIP
"""

import requests
import base64
from PIL import Image
import io
import os
import time

def test_backend_health():
    """Tester la santé du backend"""
    print("🔍 Test de santé du backend...")
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Backend healthy: {result}")
            return True
        else:
            print(f"❌ Backend unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend inaccessible: {e}")
        return False

def test_streamlit_health():
    """Tester la santé de Streamlit"""
    print("🔍 Test de santé de Streamlit...")
    try:
        response = requests.get("http://localhost:8503", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit accessible")
            return True
        else:
            print(f"❌ Streamlit inaccessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streamlit inaccessible: {e}")
        return False

def test_heatmap_with_real_image():
    """Tester la heatmap avec une vraie image"""
    print("🔍 Test de heatmap avec vraie image...")
    
    # Utiliser une vraie image du dataset
    image_path = 'Images/009099b1f6e1e8f893ec29a7023153c4.jpg'
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return False
    
    try:
        # Charger l'image
        img = Image.open(image_path)
        print(f"   Image chargée: {img.size}, mode: {img.mode}")
        
        # Convertir en base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Tester la heatmap
        data = {
            'image': img_b64,
            'text_description': 'montre analogique pour homme en acier inoxydable',
            'product_keywords': 'watch, analog, men, stainless, steel, water, resistant',
            'action': 'heatmap'
        }
        
        response = requests.post('http://localhost:5000/score', json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Heatmap générée avec succès")
            print(f"   Source: {result.get('source')}")
            print(f"   Keywords: {result.get('keywords')}")
            print(f"   Heatmap shape: {len(result.get('heatmap', []))} x {len(result.get('heatmap', [[]])[0]) if result.get('heatmap') else 'N/A'}")
            print(f"   Top keywords: {result.get('top_keywords')}")
            
            # Vérifier que la heatmap n'est pas uniforme
            heatmap = result.get('heatmap', [])
            if heatmap:
                import numpy as np
                heatmap_array = np.array(heatmap)
                print(f"   Heatmap stats: min={heatmap_array.min():.3f}, max={heatmap_array.max():.3f}, std={heatmap_array.std():.3f}")
                if heatmap_array.std() > 0.01:
                    print("✅ Heatmap variée - vraie attention CLIP!")
                    return True
                else:
                    print("⚠️ Heatmap uniforme - possible simulation")
                    return False
            else:
                print("❌ Pas de heatmap dans la réponse")
                return False
        else:
            print(f"❌ Erreur heatmap: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test heatmap: {e}")
        return False

def test_prediction():
    """Tester la prédiction"""
    print("🔍 Test de prédiction...")
    try:
        data = {
            'text': 'montre analogique pour homme',
            'product_keywords': 'watch, analog, men, stainless, steel'
        }
        
        response = requests.post('http://localhost:5000/score', json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prédiction réussie")
            print(f"   Catégorie: {result.get('predicted_category')}")
            print(f"   Confiance: {result.get('confidence')}")
            print(f"   Source: {result.get('source')}")
            return True
        else:
            print(f"❌ Erreur prédiction: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test prédiction: {e}")
        return False

def main():
    """Test complet de l'application"""
    print("🚀 Test complet de l'application avec backend CLIP")
    print("=" * 60)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Streamlit Health", test_streamlit_health),
        ("Prédiction", test_prediction),
        ("Heatmap avec vraie image", test_heatmap_with_real_image)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur inattendue: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Pause entre les tests
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("✅ L'application est prête avec de vraies heatmaps CLIP")
    else:
        print("⚠️ Certains tests ont échoué")
    
    return passed == total

if __name__ == "__main__":
    main()
