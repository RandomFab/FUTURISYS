---
title: Futurisys
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: Dockerfile
pinned: false
---

# 🚀 FUTURISYS — Déploiement d’un modèle de Machine Learning

> **Projet pédagogique** dans le cadre du parcours *Machine Learning & Data Science*.  
> Objectif : rendre un modèle de classification opérationnel et accessible via une API FastAPI.

---

## 🧠 Contexte du projet

Futurisys est une entreprise innovante souhaitant rendre ses modèles de machine learning accessibles à ses équipes via une **API performante**.  
Le but de ce projet est de **déployer un modèle de ML existant** (issu du projet 4 : *classification automatique d’informations*) à l’aide d’outils modernes d’ingénierie logicielle.

---

## 🎯 Objectifs

- Exposer le modèle via une **API FastAPI**.  
- Automatiser les tests et le déploiement (CI/CD).  
- Gérer la version du code avec **Git & GitHub**.  
- Documenter l’API et le code.  
- *(Étapes futures)* Connecter l’API à une **base de données PostgreSQL** pour la traçabilité des prédictions.

---

## 🧩 Structure du projet

```
FUTURISYS/
├── app/
│   ├── main.py                # Point d'entrée de l'API FastAPI
│   ├── model/                 # Modèle entraîné + préprocesseur
│   │   ├── model.pkl
│   │   └── preprocessing.pkl
│   ├── utils/                 # Fonctions utilitaires
│   └── tests/                 # Tests unitaires Pytest
├── notebooks/                 # Analyse exploratoire et entraînement du modèle
├── data/                      # Jeux de données (optionnel)
├── pyproject.toml             # Géré par uv
├── uv.lock
├── .python-version
├── requirements.txt           # Export pour CI/CD (*auto-généré*)
├── .github/
│   └── workflows/ci.yml       # Pipeline CI/CD (GitHub Actions)
└── README.md
```

---

## ⚙️ Installation

### 🔧 Prérequis

- [Python ≥ 3.10](https://www.python.org/downloads/) 
- Fichier `.python-version` défini pour garantir la compatibilité de l’environnement
- [uv](https://docs.astral.sh/uv/)  
- [Git](https://git-scm.com/)  
- *(optionnel)* Compte [Hugging Face](https://huggingface.co/) pour le déploiement

### 💻 Cloner le dépôt

```bash
git clone https://github.com/RandomFab/FUTURISYS.git
cd FUTURISYS
```

### 📦 Installer les dépendances

Le fichier `uv.lock` fige les versions exactes des dépendances afin d’assurer la reproductibilité de l’environnement.


Avec **uv** :

```bash
uv sync
```

## 🚀 Lancer l’application

### Exécution locale

```bash
uvicorn app.main:app --reload
```

👉 L’API sera disponible à l’adresse :  
**http://127.0.0.1:8000**

### Documentation interactive

Une fois l’API lancée :
- Interface Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Interface ReDoc : [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧠 Détails techniques du modèle

### 🔹 Description générale

- **Type de modèle :** HistGradientBoosting 
- **Nature des données :**     
      {
        "heure_supplementaires": int/bool
        "age": int
        "FE_ratio_ancienneté": float
        "FE_cadre": int
        "frequence_deplacement": int 0(peu), 1(occasionnel), 2(fréquent)
        "FE_duree_moy_exp_precedentes": float
        "FE_ratio_evolution": float
        "niveau_education": int 1,2,3,4,5
        "FE_reste_plus_longtemps": int/bool
        "poste": str 'Assistant de Direction','Cadre Commercial','Consultant','Directeur Technique','Manager','Représentant Commercial','Ressources Humaines','Senior Manager','Tech Lead',\n
        "statut_marital": str'Célibataire','Marié(e)','Divorcé(e)'\n
      }
- **Tâche supervisée :** classification binaire (reste dans l'entreprise / part de l'entreprise)
- **Objectif métier :**  Anticiper le départ d'unn collaborateur

### 🔹 Entraînement du modèle

- **Jeu de données source :** Evaluations annuelles, fichier SIRH et sondage de l'entreprise TECHNOVA
- **Prétraitements appliqués :** StandardScaler, OneHotEncoder, SMOTE, underscaling 
- **Pipeline d’entraînement :**  pipeline = IMBpipeline([
                                            ('preprocessing', preprocessor),
                                            ('smote',SMOTE(sampling_strategy=0.2,random_state=42)),
                                            ('under',RandomUnderSampler(sampling_strategy=0.8,random_state=42)),
                                            ('model', HistGradientBoostingClassifier(random_state=42))
                                          ])
  
- **Métriques principales :** Optimisation fait sur le recall pour identifier au maximum les personne quittant l'entreprise, quite à avoir plus de faux positif.

### 🔹 Sauvegarde et chargement du modèle

Le modèle et les objets associés (préprocesseur, encodeurs, etc.) sont sauvegardés avec `joblib` :

```python
import joblib

# Sauvegarde
joblib.dump(model, "app/model/model.pkl")
joblib.dump(preprocessor, "app/model/preprocessing.pkl")

# Chargement
model = joblib.load("app/model/model.pkl")
preprocessor = joblib.load("app/model/preprocessing.pkl")
```

---

## 🧪 Tests

Pour exécuter les tests unitaires :

```bash
pytest --cov=app
```

---

## ⚡️ Intégration Continue (CI)

Un pipeline CI est configuré avec **GitHub Actions** :

- Exécute les tests à chaque `push` ou `pull request` vers `main` ou `dev`.  
- Génère un rapport de couverture.  
- Prépare le terrain pour le déploiement automatique sur Hugging Face Spaces.

Badge CI (à compléter une fois le workflow actif) :

![CI](https://github.com/RandomFab/FUTURISYS/actions/workflows/ci.yml/badge.svg)

---

## 🧱 Étapes du projet

| Étape | Description | Statut |
|:------|:-------------|:--------|
| 1 | Mise en place du dépôt Git et structure du projet | ✅ |
| 2 | Configuration CI/CD (GitHub Actions, HF Spaces) | ✅ |
| 3 | Création de l’API FastAPI exposant le modèle | ✅ |
| 4 | Intégration d’une base PostgreSQL (traçabilité des prédictions) | 🔜 |
| 5 | Suite de tests unitaires et fonctionnels | 🔜 |
| 6 | Documentation complète et présentation finale | 🔜 |

---

## 🔐 Gestion des secrets (*à compléter*)

- [X] Ajouter les variables d’environnement sensibles (ex : URL de base de données, clés API).  
- [X] Configurer le stockage sécurisé sur GitHub (`Settings > Secrets and variables > Actions`).

---

## 📘 Ressources & Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)  
- [uv Documentation](https://docs.astral.sh/uv/)  
- [GitHub Actions Documentation](https://docs.github.com/en/actions)  
- [pytest Documentation](https://docs.pytest.org/en/stable/)  
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 👤 Auteur

**Nom :** RandomFab  
**Rôle :** Étudiant en Data Science 
**Contact :** https://github.com/RandomFab
