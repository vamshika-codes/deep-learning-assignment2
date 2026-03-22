# Deep Learning Assignment 2

## 📌 Overview
This assignment implements three deep learning models using **PyTorch**:
- **Part A**: CNN Image Classification on CIFAR-10
- **Part B**: RNN / LSTM / GRU for Time Series Prediction
- **Part C**: Conditional GAN (cGAN) on Fashion-MNIST

## 🛠️ Technologies Used
- Python
- PyTorch & TorchVision
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

## 📂 Files
- `deep_learning_assignment.py` — Complete assignment code (Parts A, B, C)
- `results/` — Generated plots and evaluation outputs
- `results/gan_samples/` — cGAN generated image samples per epoch

## ▶️ How to Run

1. Clone the repository
   git clone https://github.com/vamshika-codes/deep-learning-assignment2.git

2. Install dependencies
   pip install torch torchvision numpy matplotlib seaborn scikit-learn

3. Run the script
   python deep_learning_assignment.py

## 📊 Models

### Part A — CNN (CIFAR-10)
- Custom CNN with 3 convolutional blocks
- Transfer Learning using ResNet18 (fine-tuned)
- Techniques: BatchNorm, Dropout, CosineAnnealingLR, Label Smoothing

### Part B — RNN/LSTM/GRU
- Three sequence models compared using RMSE
- Applied to time series forecasting

### Part C — Conditional GAN (Fashion-MNIST)
- Generator + Discriminator with label embeddings
- Stability techniques: label smoothing, 2× Generator updates, LR decay

## 👩‍💻 Author
Vamshika U Devadiga
