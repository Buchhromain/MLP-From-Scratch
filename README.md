
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
| Confusion matrix                 | 🚧 (in progress) |
| Loss tracking and visualization | ✅           |
| Low-level API implementation    | 🚧 (in progress) |
| REST API Web Interface          | 🚧 (in progress) |
| Multithread support             | 🚧 (optional) |

---

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
node mlp.js
```

---

## 📊 Expected Outputs

- ✅ Trained model saved to disk
- ✅ Confusion matrix displayed
- ✅ Loss curve generated (`loss_plot.html`)
- ✅ Evaluation on test set

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

**Romain Buchheister**  
> A passionate learner — this project demonstrates my knowledge and hands-on capability in designing neural networks from scratch.
