#!/bin/bash
# =========================================================
# Script : create_env.sh
# Objectif : Recréer un environnement Python isolé pour FED_PROJECT
# Usage : bash create_env.sh
# =========================================================

# 1. Se placer à la racine du projet
cd "$(dirname "$0")"
echo "[INFO] Position actuelle : $(pwd)"

# 2. Supprimer tout ancien environnement
echo "[INFO] Suppression de l'ancien venv..."
rm -rf .venv

# 3. Créer un nouvel environnement (Python 3.x)
echo "[INFO] Création du nouvel environnement virtuel..."
python3 -m venv .venv

# 4. Activer l'environnement
source .venv/bin/activate

# 5. Mettre pip à jour
echo "[INFO] Mise à jour de pip..."
pip install --upgrade pip

# 6. Installer les dépendances
echo "[INFO] Installation des dépendances..."
pip install -r requirements.txt

# 7. Installation du projet local en mode editable
echo "[INFO] Installation du package local src/..."
pip install -e .

# 8. Créer le kernel Jupyter associé
echo "[INFO] Création du kernel Jupyter..."
python -m ipykernel install --user --name=fed_env --display-name "Python (fed_env)"

echo "[SUCCESS] Environnement 'fed_env' prêt à l'emploi."
