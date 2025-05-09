
# 🧠 MLP From Scratch

Ce projet implémente un **Perceptron Multi-Couches (MLP)** **sans utiliser de frameworks de deep learning** comme TensorFlow ou PyTorch. Tout a été codé from scratch pour permettre une compréhension complète des mécanismes internes.

---

## 📚 Objectif

Créer une API d'entraînement et d'inférence pour un réseau de neurones MLP, avec les contraintes suivantes :

- ⚠️ Aucun framework AI (PyTorch, TensorFlow, etc.)
- 🔁 Architecture MLP flexible
- 🧠 Multiples fonctions d'activation
- 📊 Classification et régression
- 🧮 Initialisation des poids paramétrable
- 🏃 Optimiseurs SGD : Momentum, RMSProp, Adam
- 🛑 Critères d’arrêt personnalisables
- 🧽 Régularisation L1, L2, Elastic Net
- 📉 Suivi de la perte
- ✅ Matrice de confusion
- 🔀 (Optionnel) Exécution parallèle (multi-thread/GPU)
- 🔌 (Optionnel) API Web REST
- 🔧 Helpers et structure modulaire

---

## 🛠️ Fonctionnalités implémentées

| Fonctionnalité                    | Statut      |
|----------------------------------|-------------|
| Architecture MLP modulaire       | ✅           |
| Fonctions d'activation (ReLU, Sigmoid, Tanh, etc.) | ✅ |
| Classification (MNIST)          | ✅           |
| Régression (Boston Housing)     | ✅           |
| Initialisation Xavier / He / Aléatoire | ✅    |
| Optimiseurs : SGD, Momentum, RMSProp, Adam | ✅  |
| Critères d'arrêt (convergence, patience, epochs) | ✅ |
| Régularisation : L1, L2, ElasticNet | ✅         |
| Matrice de confusion             | ✅           |
| Suivi de la perte et visualisation | ✅         |
| API Python bas-niveau            | ✅           |
| API Web REST                     | 🚧 (en cours)|
| Exécution multithread            | 🚧 (optionnel) |

---

## 📁 Structure du projet

```
mlp-from-scratch/
│
├── src/                     # Code source principal
│   ├── core/                # MLP, layers, loss, activation
│   ├── optim/               # Implémentation des optimiseurs
│   ├── utils/               # Fonctions utilitaires
│   └── api/                 # Interface d'entraînement et d'inférence
│
├── data/                    # Jeux de données CSV
│
├── results/                 # Modèles entraînés, courbes, matrices
│
├── mnist-in-csv/            # Dataset MNIST en format CSV
│
├── mlp.js                   # Fichier principal d'exécution
├── loss_plot.html           # Visualisation de la courbe de perte
└── README.md                # Ce fichier
```

---

## 📦 Installation

```bash
git clone https://github.com/Buchhromain/MLP-From-Scratch.git
cd MLP-From-Scratch
npm install
```

> ⚠️ Si ton projet est en JS (Node.js), adapte selon le runtime (ou `python` si tu es passé par un autre langage).

---

## 🚀 Lancer un entraînement (exemple)

```bash
node mlp.js --mode train --dataset mnist-in-csv/mnist_train.csv
```

---

## 📊 Résultats attendus

- ✅ Sauvegarde du modèle entraîné
- ✅ Matrice de confusion affichée
- ✅ Courbe de perte générée (`loss_plot.html`)
- ✅ Évaluation sur test set
- ✅ Fichier `model.json` sauvegardé

---

## 📈 Exemple de visualisation

*(à ajouter plus tard : image ou capture de ta courbe de perte ou de la matrice de confusion)*

---

## 📄 Sources de données

- [MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- [Boston Housing](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)

---

## 🧠 Conçu sans frameworks ML

Ce projet est entièrement fait maison pour garantir une **compréhension complète du processus de deep learning**, de la propagation avant au backpropagation.

---

## 📌 À faire

- [ ] API Web REST (Express ou FastAPI)
- [ ] Support GPU avec WebGL ou CUDA (optionnel)
- [ ] Interface utilisateur simple

---

## 🙋‍♂️ Auteur

**Romain "Buchhromain"**  
> Étudiant passionné, ce projet est une démonstration de compréhension et de savoir-faire algorithmique autour des réseaux de neurones.
