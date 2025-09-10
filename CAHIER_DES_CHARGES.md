# 📋 CAHIER DES CHARGES - PROJET CLASSIFICATION PRODUITS

## 🎯 OBJECTIF PRINCIPAL
**Déployer le modèle PyTorch finetuné dans Azure et lier le reste de l'application à ça pour la prédiction. Supprimer ONNX de Azure.**

## ✅ EXIGENCES FONCTIONNELLES

### 1. **Modèle**
- ✅ **PyTorch uniquement** - Modèle CLIP finetuné
- ❌ **ONNX interdit** - Supprimer complètement d'Azure
- ✅ **Modèle réel** - Pas de simulation, pas de résultats simulés

### 2. **Déploiement**
- ✅ **100% Cloud** - Application entièrement sur Azure ML + Streamlit Cloud
- ❌ **Local interdit** - Aucun composant local
- ✅ **Endpoint Azure ML** - Modèle PyTorch déployé sur Azure ML

### 3. **Application**
- ✅ **Streamlit Cloud** - Interface utilisateur sur Streamlit Cloud
- ✅ **Prédiction de catégories** - Classification de produits
- ✅ **Images + texte** - Prise en charge des images et descriptions

## 🚫 CONTRAINTES STRICTES

### **INTERDICTIONS ABSOLUES**
1. ❌ **ONNX** - Aucune référence, aucun modèle, aucun message
2. ❌ **Localhost** - Aucun serveur local, aucune référence localhost
3. ❌ **Simulation** - Aucun résultat simulé ou factice
4. ❌ **Dispersion** - Pas de fonctionnalités annexes (heatmaps, attention, etc.)

### **OBLIGATIONS**
1. ✅ **PyTorch uniquement** - Modèle CLIP finetuné
2. ✅ **Azure ML** - Endpoint cloud Azure ML
3. ✅ **Streamlit Cloud** - Application web cloud
4. ✅ **Messages cohérents** - Tous les messages mentionnent PyTorch

## 📊 ÉTAT ACTUEL

### ✅ **RÉALISÉ**
- Migration ONNX → PyTorch terminée
- Messages corrigés (plus de référence ONNX)
- Dépendances Streamlit Cloud corrigées
- Erreurs d'indentation corrigées
- Endpoint Azure ML restauré

### ❌ **PROBLÈME IDENTIFIÉ**
- Endpoint Azure ML ne répond plus : `https://new-clip-classification-app-hrgracfqaegbd9ek.francecentral-01.azurewebsites.net/score`
- Erreur d'application sur l'endpoint

## 🎯 PROCHAINES ACTIONS OBLIGATOIRES

### **PRIORITÉ 1 : RÉPARER L'ENDPOINT AZURE ML**
1. Vérifier le statut de l'endpoint Azure ML
2. Redéployer le modèle PyTorch si nécessaire
3. Tester la connectivité

### **PRIORITÉ 2 : VALIDATION FINALE**
1. Tester la prédiction end-to-end
2. Vérifier que c'est bien PyTorch (pas ONNX)
3. Confirmer que c'est 100% cloud

## 📝 RÈGLES DE CONDUITE

### **AVANT CHAQUE ACTION**
1. ✅ Lire ce cahier des charges
2. ✅ Vérifier que l'action respecte les contraintes
3. ✅ Ne pas ajouter de fonctionnalités non demandées
4. ✅ Se concentrer uniquement sur l'objectif principal

### **INTERDICTIONS**
- ❌ Ne pas créer de scripts locaux
- ❌ Ne pas ajouter de fonctionnalités annexes
- ❌ Ne pas mentionner ONNX
- ❌ Ne pas utiliser localhost
- ❌ Ne pas se disperser

## 🎯 OBJECTIF FINAL
**Application Streamlit Cloud qui utilise un modèle PyTorch finetuné déployé sur Azure ML pour la classification de produits, sans aucune référence à ONNX ou composants locaux.**

---
**Date de création :** 10 septembre 2025  
**Statut :** En cours - Endpoint Azure ML à réparer  
**Prochaine action :** Réparer l'endpoint Azure ML
