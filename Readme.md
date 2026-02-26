# Pipeline de Traduction Vidéo Multilingue (mTEDx)

Ce projet implémente une chaîne de traitement complète basée sur le Deep Learning pour la transcription et la traduction de vidéos TEDx françaises vers 5 langues cibles (Anglais, Espagnol, Portugais, Arabe et Allemand).

## 🚀 Installation et Configuration

Pour cloner et configurer l'environnement de développement :

1. **Clonage du dépôt** :
   git clone https://github.com/CoverilleCodeGit00/Project_DL_Translation.git
   
   cd Projet_DL_Translation

2. **Création de l'environnement virtuel :** :
    python -m venv venv_dl

3. **Activation de l'environnement :** :
    
    Windows : venv_dl\Scripts\activate
    
    Linux/Mac : source venv_dl/bin/activate

4. **Installation des dépendances** :
    pip install -r requirements.txt

## 📂 Structure du Projet

L'architecture logicielle est organisée de manière modulaire pour séparer les données, le code source et les expérimentations :

data/ : Gestion des données (fichiers volumineux ignorés par Git).

    raw/ : Stockage des archives originales .tgz (mtedx_fr.tgz, etc.).

    processed/ : Index CSV générés et futurs segments audio extraits (16kHz).

    temp/ : Zone de transit pour les opérations d'extraction temporaires.

src/ : Code source Python contenant les classes et fonctions modulaires.

notebooks/ : Journaux d'expérimentation (Indexation, ASR, NMT).

models/ : Répertoire de sauvegarde des poids des modèles entraînés (.pt).

outputs/ : Résultats finaux (fichiers de sous-titres .SRT, rapports).

Note : Les dossiers vides contiennent un fichier .gitkeep pour maintenir l'arborescence sur le dépôt distant sans inclure les fichiers lourds.

## 📊 État Actuel du Projet

[x] Phase 1 : Indexation : Master Index multilingue terminé et séparation des index ASR (FR) et NMT (paires de langues).

[ ] Phase 2 : Traitement Audio : Extraction physique des segments audio (16kHz) et évaluation Whisper.

[ ] Phase 3 : Modélisation NMT : Implémentation de la baseline LSTM et des modèles SOTA (MarianMT, NLLB-200).

[ ] Phase 4 : Évaluation & Livrables : Analyse des métriques (WER, BLEU) et génération des fichiers SRT finaux.