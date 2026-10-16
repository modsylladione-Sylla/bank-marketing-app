# bank-marketing-app
# 🏦 Prédiction de souscription à un dépôt à terme

Mini-projet de data science : prédire si un client d'une banque va souscrire à un **dépôt à terme** après un appel de télémarketing, puis exposer le modèle dans une application web Streamlit.

**Application en ligne** : https://bank-marketing-app-jtcabjutz8sv9xo6stg9e2.streamlit.app/

## Résumé

Avec un seuil de décision de **0,28**, le modèle identifie environ **47 %** des clients qui souscrivent, et un client signalé sur deux souscrit réellement. Sur le jeu de test, le coût métier est réduit d'environ **24 %** par rapport à la meilleure stratégie naïve (n'appeler personne). Le gain est modeste parce que la durée de l'appel, variable très prédictive mais indisponible avant l'appel, est volontairement exclue.

## Données

- Jeu **Bank Marketing** de l'UCI Machine Learning Repository (`bank-additional-full.csv`) : 41 188 appels, 20 variables explicatives.
- Cible : `y = yes` si le client a souscrit un dépôt à terme. La classe est minoritaire (environ 11 % des appels).
- Variables : profil du client (âge, profession, situation maritale, éducation…), historique de campagne (mois, nombre de contacts, résultat précédent…) et indicateurs macro-économiques (Euribor 3 mois, indice de confiance…).

## Méthode

1. **Exploration** : valeurs manquantes, répartition de la cible, modalités `unknown`.
2. **Pipeline scikit-learn** : `StandardScaler` sur les variables numériques, `OneHotEncoder` sur les catégories, régression logistique. Le tout est ajusté sur le jeu d'entraînement uniquement, sans fuite de données.
3. **Séparation** 80 % / 20 %, stratifiée sur la cible.
4. **Choix du modèle et du seuil** par validation croisée (5 plis) sur le jeu d'entraînement : régression logistique standard contre `class_weight="balanced"`. Le seuil retenu minimise un coût métier (appel inutile = 500, souscription manquée = 1 000, valeurs hypothétiques à adapter).
5. **Évaluation finale** sur le jeu de test, utilisé une seule fois : AUC, rappel, précision, matrice de confusion et coût comparé à deux stratégies naïves.
6. **Sauvegarde** d'un fichier unique `bank_model.joblib` (pipeline, seuil, métadonnées du formulaire), réentraîné sur toutes les données avant déploiement.

### Pourquoi `duration` est exclue

La durée de l'appel n'est connue qu'**après** l'appel. Elle améliore fortement les scores, mais un modèle qui l'utilise ne peut pas servir à décider qui appeler. Le notebook la retire par défaut (`USE_DURATION = False`) et compare les résultats avec et sans elle.

## Résultats (jeu de test)

Sensibilité au seuil, avec le seuil retenu en gras :

| Seuil | Rappel (oui) | Précision (oui) | Accuracy | Coût |
|---|---|---|---|---|
| 0,20 | 0,595 | 0,452 | 0,873 | 711 000 |
| **0,28** | **0,472** | **0,508** | **0,889** | **702 500** |
| 0,30 | 0,439 | 0,508 | 0,889 | 718 000 |
| 0,40 | 0,319 | 0,591 | 0,898 | 734 500 |

Comparaison avec des stratégies naïves (jeu de test : 928 souscripteurs, 7 310 non-souscripteurs) :

| Stratégie | Coût |
|---|---|
| Appeler tout le monde | 3 655 000 |
| N'appeler personne | 928 000 |
| **Modèle, seuil 0,28** | **702 500** |


**Lecture**
- Le coût varie peu entre 0,20 et 0,40 (moins de 5 % d'écart) : le choix du seuil est peu sensible, l'optimum se situe entre 0,2 et 0,3.
- L'accuracy augmente avec le seuil alors que le rappel s'effondre. Elle est peu informative ici : répondre « non » à tout le monde donne déjà environ 0,89.
- Le seuil de 0,28 est proche du seuil théorique 500 / (500 + 1 000) ≈ 0,33 pour un modèle bien calibré.

## Structure du dépôt

```
├── Banque_Mini_projet_corrige.ipynb   # analyse, modélisation, sauvegarde du modèle
├── app.py                             # application Streamlit
├── bank_model.joblib                  # pipeline entraîné, seuil, métadonnées
├── requirements.txt                   # dépendances
└── README.md
```

## Lancer le projet en local

```bash
git clone https://github.com/modsylladione-Sylla/bank-marketing-app.git
cd TON-DEPOT
pip install -r requirements.txt
streamlit run app.py
```

Pour regénérer le modèle, exécute le notebook de haut en bas : il télécharge les données (ou lit `bank-additional-full.csv` s'il est dans le dossier) et écrit `bank_model.joblib`.

> Utilise la **même version de scikit-learn** pour l'entraînement et pour l'application, sinon le chargement du modèle peut échouer.

## Limites

- Les indicateurs macro-économiques (`emp.var.rate`, `euribor3m`, `nr.employed`) sont fortement corrélés : leurs coefficients ne s'interprètent pas isolément.
- Les coefficients décrivent des associations, pas des causes.
- Les données couvrent une période précise (2008-2010) et une banque portugaise : le modèle ne se transpose pas tel quel à un autre contexte.
- Les coûts utilisés pour choisir le seuil sont des hypothèses ; avec d'autres coûts, le seuil optimal change.

## Auteur

Modou Dione

Données : S. Moro, P. Cortez, P. Rita, *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems, 2014.
