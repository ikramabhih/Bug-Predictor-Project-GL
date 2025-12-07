Bug Predictor Project – Génie Logiciel & IA

Ce projet a pour objectif de prédire automatiquement la présence de bugs dans des fichiers de code source à l’aide de techniques de Machine Learning.
Il est réalisé dans le cadre du module Génie Logiciel Master S3.

Objectifs du Projet

Extraire automatiquement des métriques de code
(complexité cyclomatique, lignes de code, classes, fonctions, imports, imbrication, etc.)

Entraîner plusieurs modèles de Machine Learning :

    Régression Logistique

    Random Forest

    XGBoost

Comparer les performances des modèles pour sélectionner le meilleur.

Interpréter les résultats et analyser les métriques les plus déterminantes.

Déployer une API Flask permettant :

  d’envoyer du code Python

  d’obtenir automatiquement la prédiction « bug » ou « pas bug »

Créer une interface Web (HTML / CSS / JS) permettant plusieurs actions :

    Entrer des métriques manuellement

    Uploader un fichier pour analyse

    Copier-coller du code directement dans un textarea

    Analyser automatiquement un repository GitHub
    (cloner un repo → extraire les métriques → prédire les bugs)
