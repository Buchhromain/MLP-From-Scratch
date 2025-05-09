
# 🧠 MLP From Scratch

This project implements a **Multi-Layer Perceptron (MLP)** **without using any deep learning frameworks** such as TensorFlow or PyTorch. Everything is built from the ground up to ensure a complete understanding of internal mechanisms.

---

## 📚 Objective

Build a training and inference API for an MLP neural network, fulfilling the following requirements:

- ⚠️ No AI frameworks (e.g., PyTorch, TensorFlow, etc.)
- 🔁 Flexible MLP architecture
- 🧠 Multiple activation functions
- 📊 Support for classification and regression
- 🧮 Customizable weight initialization
- 🏃 SGD optimizers: Momentum, RMSProp, Adam
- 🛑 Customizable stopping criteria
- 🧽 L1, L2, and Elastic Net regularization
- 📉 Loss tracking and visualization
- ✅ Confusion matrix computation
- 🔀 (Optional) Parallel training (multi-thread/GPU)
- 🔌 (Optional) Web API interface
- 🔧 Helper classes and modular structure

---

## 🛠️ Implemented Features

| Feature                          | Status      |
|----------------------------------|-------------|
| Modular MLP architecture         | ✅           |
| Activation functions (ReLU, Sigmoid, Tanh, etc.) | ✅ |
| Classification (MNIST)          | ✅           |
| Regression (Boston Housing)     | ✅           |
| Xavier / He / Random initialization | ✅        |
| Optimizers: SGD, Momentum, RMSProp, Adam | ✅     |
| Stopping criteria (convergence, patience, epochs) | ✅ |
| Regularization: L1, L2, ElasticNet | ✅         |
| Confusion matrix                 | ✅           |
| Loss tracking and visualization | ✅           |
| Low-level API implementation    | ✅           |
| REST API Web Interface          | 🚧 (in progress) |
| Multithread support             | 🚧 (optional) |

---

## 📁 Project Structure

```
mlp-from-scratch/
│
├── src/                     # Core source code
│   ├── core/                # MLP, layers, loss, activations
│   ├── optim/               # Optimizer implementations
│   ├── utils/               # Utility functions
│   └── api/                 # Training/inference interface
│
├── data/                    # CSV datasets
│
├── results/                 # Trained models, plots, confusion matrices
│
├── mnist-in-csv/            # MNIST dataset in CSV format
│
├── mlp.js                   # Main entry point
├── loss_plot.html           # Loss curve visualization
└── README.md                # This file
```

---

## 📦 Installation

```bash
git clone https://github.com/Buchhromain/MLP-From-Scratch.git
cd MLP-From-Scratch
npm install
```

> ⚠️ If your project is written in JS (Node.js), adapt as needed (or use `python` if built in another language).

---

## 🚀 Run Training (Example)

```bash
node mlp.js --mode train --dataset mnist-in-csv/mnist_train.csv
```

---

## 📊 Expected Outputs

- ✅ Trained model saved to disk
- ✅ Confusion matrix displayed
- ✅ Loss curve generated (`loss_plot.html`)
- ✅ Evaluation on test set
- ✅ Model saved as `model.json`

---

## 📈 Example Visualization

*(To be added later: image of loss curve or confusion matrix screenshot)*

---

## 📄 Datasets

- [MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- [Boston Housing](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)

---

## 🧠 Framework-Free Design

This project is entirely handcrafted to provide a **deep understanding of deep learning** — from forward propagation to backpropagation.

---

## 📌 To Do

- [ ] REST API Interface (e.g. Express or FastAPI)
- [ ] GPU support using WebGL or CUDA (optional)
- [ ] Simple user interface

---

## 🙋‍♂️ Author

**Romain "Buchhromain"**  
> A passionate learner — this project demonstrates my knowledge and hands-on capability in designing neural networks from scratch.
