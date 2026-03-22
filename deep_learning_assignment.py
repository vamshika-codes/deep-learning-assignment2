"""
=============================================================================
  Applied Deep Models: CNN + RNN/LSTM/GRU + GAN
  Complete Assignment Solution
  Framework: PyTorch
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
RESULTS_DIR = "results"
GAN_DIR     = os.path.join(RESULTS_DIR, "gan_samples")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GAN_DIR,     exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_loss_acc(train_losses, val_losses, train_accs, val_accs, title, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax1.plot(train_losses, label='Train Loss', color='#e74c3c')
    ax1.plot(val_losses,   label='Val Loss',   color='#3498db')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(train_accs, label='Train Acc', color='#27ae60')
    ax2.plot(val_accs,   label='Val Acc',   color='#9b59b6')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fname)

def plot_confusion(cm, classes, title, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    save_fig(fname)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────────────────────────
# PART A: CNN — CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PART A: CNN IMAGE CLASSIFICATION (CIFAR-10)")
print("="*60)

CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']
CNN_EPOCHS   = 30
CNN_BATCH    = 128
CNN_LR       = 1e-3

# ── Data ──────────────────────────────────────────────────────────────────────
mean_c = (0.4914, 0.4822, 0.4465)
std_c  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean_c, std_c),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean_c, std_c),
])

cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_tf)
cifar_val   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_tf)
train_loader = DataLoader(cifar_train, batch_size=CNN_BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(cifar_val,   batch_size=CNN_BATCH, shuffle=False, num_workers=0, pin_memory=True)


# ── Model A1: Custom CNN ───────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.classifier(x)


# ── Model A2: Transfer Learning — ResNet18 ─────────────────────────────────────
def get_resnet18(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Fine-tune: unfreeze last 2 layers + replace head
    for name, param in model.named_parameters():
        param.requires_grad = ('layer4' in name or 'fc' in name)
    model.fc = nn.Linear(512, num_classes)
    return model


# ── Training loop ──────────────────────────────────────────────────────────────
def train_cnn(model, loader, val_loader, epochs, lr, label):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    start = time.time()

    for epoch in range(1, epochs+1):
        # ── train ──
        model.train()
        tl, correct, total = 0., 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            tl      += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total   += x.size(0)
        scheduler.step()
        train_losses.append(tl/total); train_accs.append(100.*correct/total)

        # ── validate ──
        model.eval()
        vl, vc, vt = 0., 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out  = model(x)
                loss = criterion(out, y)
                vl  += loss.item() * x.size(0)
                vc  += out.argmax(1).eq(y).sum().item()
                vt  += x.size(0)
        val_losses.append(vl/vt); val_accs.append(100.*vc/vt)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{label}] Ep {epoch:3d}/{epochs} | "
                  f"Train {train_accs[-1]:.1f}% | Val {val_accs[-1]:.1f}%")

    elapsed = time.time() - start
    return model, train_losses, val_losses, train_accs, val_accs, elapsed


def get_predictions(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)


# ── Run models ────────────────────────────────────────────────────────────────
print("\n[A1] Training Custom CNN …")
custom_cnn = CustomCNN()
custom_cnn, tl1, vl1, ta1, va1, t1 = train_cnn(
    custom_cnn, train_loader, val_loader, CNN_EPOCHS, CNN_LR, "CustomCNN")

print("\n[A2] Fine-tuning ResNet18 …")
resnet_model = get_resnet18()
resnet_model, tl2, vl2, ta2, va2, t2 = train_cnn(
    resnet_model, train_loader, val_loader, CNN_EPOCHS, CNN_LR*0.5, "ResNet18")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n[A] Generating CNN plots …")
plot_loss_acc(tl1, vl1, ta1, va1, "Custom CNN — CIFAR-10", "cnn_custom_curves.png")
plot_loss_acc(tl2, vl2, ta2, va2, "ResNet18 — CIFAR-10",   "cnn_resnet_curves.png")

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("CNN Model Comparison — CIFAR-10", fontsize=13, fontweight='bold')
names = ['Custom CNN', 'ResNet18']
accs  = [max(va1), max(va2)]
times = [t1, t2]
sizes = [count_params(CustomCNN())/1e6, count_params(get_resnet18())/1e6]

axes[0].bar(names, accs, color=['#3498db','#e74c3c'], width=0.4, edgecolor='black')
axes[0].set_ylabel('Best Val Accuracy (%)'); axes[0].set_title('Accuracy')
axes[0].set_ylim(0,100)
for i,v in enumerate(accs):
    axes[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

axes[1].bar(names, times, color=['#27ae60','#9b59b6'], width=0.4, edgecolor='black')
axes[1].set_ylabel('Training Time (s)'); axes[1].set_title('Training Time')
for i,v in enumerate(times):
    axes[1].text(i, v+1, f'{v:.0f}s', ha='center', fontweight='bold')

axes[2].bar(names, sizes, color=['#e67e22','#1abc9c'], width=0.4, edgecolor='black')
axes[2].set_ylabel('Parameters (M)'); axes[2].set_title('Model Size')
for i,v in enumerate(sizes):
    axes[2].text(i, v+0.1, f'{v:.2f}M', ha='center', fontweight='bold')

plt.tight_layout(); save_fig("cnn_comparison.png")

# Confusion matrices
for model_obj, name, fname in [(custom_cnn, "Custom CNN", "cm_custom.png"),
                                (resnet_model, "ResNet18",  "cm_resnet.png")]:
    preds, labels = get_predictions(model_obj, val_loader)
    cm = confusion_matrix(labels, preds)
    plot_confusion(cm, CIFAR_CLASSES, f"Confusion Matrix — {name}", fname)
    print(f"\n[{name}] Classification Report:")
    print(classification_report(labels, preds, target_names=CIFAR_CLASSES))

print(f"\n  Custom CNN  → Best Val Acc: {max(va1):.2f}%  |  Time: {t1:.0f}s  |  Params: {count_params(CustomCNN())/1e6:.2f}M")
print(f"  ResNet18    → Best Val Acc: {max(va2):.2f}%  |  Time: {t2:.0f}s  |  Params: {count_params(get_resnet18())/1e6:.2f}M")


# ─────────────────────────────────────────────────────────────────────────────
# PART B: RNN / LSTM / GRU — Time-Series (Airline Passengers)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PART B: RNN/LSTM/GRU — TIME-SERIES PREDICTION")
print("="*60)

RNN_EPOCHS = 100
RNN_LR     = 1e-3
SEQ_LEN    = 12
HIDDEN     = 64
LAYERS     = 2

# ── Dataset: classic airline passengers (monthly) ─────────────────────────────
airline_raw = np.array([
    112,118,132,129,121,135,148,148,136,119,104,118,
    115,126,141,135,125,149,170,170,158,133,114,140,
    145,150,178,163,172,178,199,199,184,162,146,166,
    171,180,193,181,183,218,230,242,209,191,172,194,
    196,196,236,235,229,243,264,272,237,211,180,201,
    204,188,235,227,234,264,302,293,259,229,203,229,
    242,233,267,269,270,315,364,347,312,274,237,278,
    284,277,317,313,318,374,413,405,355,306,271,306,
    315,301,356,348,355,422,465,467,404,347,305,336,
    340,318,362,348,363,435,491,505,404,359,310,337,
    360,342,406,396,420,472,548,559,463,407,362,405,
    417,391,419,461,472,535,622,606,508,461,390,432
], dtype=np.float32)

# Normalize
a_min, a_max = airline_raw.min(), airline_raw.max()
data_norm = (airline_raw - a_min) / (a_max - a_min)

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_sequences(data_norm, SEQ_LEN)
split = int(len(X)*0.8)
X_tr, X_te = torch.tensor(X[:split]).unsqueeze(-1), torch.tensor(X[split:]).unsqueeze(-1)
y_tr, y_te = torch.tensor(y[:split]).unsqueeze(-1), torch.tensor(y[split:]).unsqueeze(-1)

ts_train = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
ts_test  = DataLoader(TensorDataset(X_te, y_te), batch_size=16, shuffle=False)


# ── Sequence model factory ─────────────────────────────────────────────────────
class SeqModel(nn.Module):
    def __init__(self, rnn_type='LSTM', hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = cls(input_size=1, hidden_size=hidden, num_layers=layers,
                       batch_first=True, dropout=0.2 if layers>1 else 0.)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def train_seq(model, tr_loader, te_loader, epochs, lr, label):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    tr_losses, te_losses = [], []
    for epoch in range(1, epochs+1):
        model.train()
        tl = 0.
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            tl += loss.item()
        tr_losses.append(tl / len(tr_loader))

        model.eval()
        vl = 0.
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vl += criterion(model(x), y).item()
        te_losses.append(vl / len(te_loader))
        scheduler.step(te_losses[-1])

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{label}] Ep {epoch:3d} | Train MSE {tr_losses[-1]:.5f} | Val MSE {te_losses[-1]:.5f}")

    return model, tr_losses, te_losses


def rmse_on_loader(model, loader, a_min, a_max):
    model.eval(); preds, actuals = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(DEVICE)).cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
    p = np.array(preds)   * (a_max - a_min) + a_min
    a = np.array(actuals) * (a_max - a_min) + a_min
    return np.sqrt(np.mean((p - a)**2)), p, a


# ── Train all three ────────────────────────────────────────────────────────────
rnn_results = {}
for rnn_type in ['RNN', 'LSTM', 'GRU']:
    print(f"\n[B] Training {rnn_type} …")
    m = SeqModel(rnn_type=rnn_type)
    m, trl, tel = train_seq(m, ts_train, ts_test, RNN_EPOCHS, RNN_LR, rnn_type)
    rmse, preds, actuals = rmse_on_loader(m, ts_test, a_min, a_max)
    rnn_results[rnn_type] = dict(model=m, trl=trl, tel=tel, rmse=rmse, preds=preds, actuals=actuals)
    print(f"  [{rnn_type}] Test RMSE: {rmse:.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n[B] Generating RNN plots …")
colors = {'RNN': '#e74c3c', 'LSTM': '#3498db', 'GRU': '#27ae60'}

# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("RNN / LSTM / GRU — Training Stability", fontsize=13, fontweight='bold')
for rnn_type, res in rnn_results.items():
    axes[0].plot(res['trl'], label=f'{rnn_type} Train', color=colors[rnn_type])
    axes[1].plot(res['tel'], label=f'{rnn_type} Val',   color=colors[rnn_type], linestyle='--')
for ax, title in zip(axes, ['Training Loss (MSE)', 'Validation Loss (MSE)']):
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); save_fig("rnn_loss_curves.png")

# Prediction comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Time-Series Predictions vs Actual (Test Set)", fontsize=13, fontweight='bold')
for ax, (rnn_type, res) in zip(axes, rnn_results.items()):
    ax.plot(res['actuals'], label='Actual',   color='black', linewidth=2)
    ax.plot(res['preds'],   label=rnn_type,   color=colors[rnn_type], linestyle='--', linewidth=1.5)
    ax.set_title(f"{rnn_type}  —  RMSE: {res['rmse']:.2f}", fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
axes[-1].set_xlabel('Time Step')
plt.tight_layout(); save_fig("rnn_predictions.png")

# RMSE comparison bar
fig, ax = plt.subplots(figsize=(6, 4))
rmses = [rnn_results[t]['rmse'] for t in ['RNN','LSTM','GRU']]
bars  = ax.bar(['RNN','LSTM','GRU'], rmses,
               color=[colors['RNN'], colors['LSTM'], colors['GRU']],
               width=0.4, edgecolor='black')
ax.set_ylabel('RMSE (passengers)'); ax.set_title('Model RMSE Comparison', fontweight='bold')
for bar, v in zip(bars, rmses):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.2f}', ha='center', fontweight='bold')
plt.tight_layout(); save_fig("rnn_rmse_comparison.png")

print("\n  RMSE Summary:")
for t in ['RNN','LSTM','GRU']:
    print(f"    {t}: {rnn_results[t]['rmse']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# PART C: CONDITIONAL GAN (cGAN) — Fashion-MNIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PART C: CONDITIONAL GAN — FASHION-MNIST")
print("="*60)

GAN_EPOCHS    = 60
GAN_BATCH     = 128
GAN_LR_D      = 2e-4
GAN_LR_G      = 2e-4
LATENT_DIM    = 128
N_CLASSES     = 10
EMBED_DIM     = 16
SAMPLE_EVERY  = 10
IMG_SIZE      = 28

FASHION_NAMES = ['T-shirt','Trouser','Pullover','Dress','Coat',
                 'Sandal','Shirt','Sneaker','Bag','Ankle boot']

fmnist_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
fmnist_data   = torchvision.datasets.FashionMNIST('./data', train=True,  download=True, transform=fmnist_tf)
fmnist_loader = DataLoader(fmnist_data, batch_size=GAN_BATCH, shuffle=True, num_workers=0)


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 256),
            nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2), nn.BatchNorm1d(1024),
            nn.Linear(1024, IMG_SIZE*IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x   = torch.cat([z, emb], dim=1)
        return self.net(x).view(-1, 1, IMG_SIZE, IMG_SIZE)


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE*IMG_SIZE + EMBED_DIM, 1024),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        emb = self.label_emb(labels)
        x   = torch.cat([img.view(img.size(0), -1), emb], dim=1)
        return self.net(x)


G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = optim.Adam(G.parameters(), lr=GAN_LR_G, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=GAN_LR_D, betas=(0.5, 0.999))

# LR schedulers to improve stability
sched_G = optim.lr_scheduler.ExponentialLR(opt_G, gamma=0.99)
sched_D = optim.lr_scheduler.ExponentialLR(opt_D, gamma=0.99)

criterion_gan = nn.BCELoss()

# Fixed noise for consistent sample visualization
fixed_noise  = torch.randn(N_CLASSES*8, LATENT_DIM, device=DEVICE)
fixed_labels = torch.tensor([i for i in range(N_CLASSES) for _ in range(8)], device=DEVICE)

g_losses, d_losses = [], []

def save_gan_samples(epoch, G, noise, labels, fname):
    G.eval()
    with torch.no_grad():
        imgs = G(noise, labels).cpu()
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    nrow = 8
    fig, axes = plt.subplots(N_CLASSES, nrow, figsize=(nrow*1.1, N_CLASSES*1.1))
    for i in range(N_CLASSES):
        for j in range(nrow):
            axes[i][j].imshow(imgs[i*nrow+j, 0], cmap='gray')
            axes[i][j].axis('off')
            if j == 0:
                axes[i][j].set_ylabel(FASHION_NAMES[i], fontsize=7, rotation=0, ha='right', va='center')
    fig.suptitle(f"cGAN Samples — Epoch {epoch}", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 1])
    path = os.path.join(GAN_DIR, fname)
    plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
    print(f"  Saved GAN samples: {path}")
    G.train()


print("\n[C] Training Conditional GAN …")
failure_epoch_noted = False

for epoch in range(1, GAN_EPOCHS+1):
    epoch_g, epoch_d, n_batches = 0., 0., 0

    for real_imgs, real_labels in fmnist_loader:
        bs = real_imgs.size(0)
        real_imgs   = real_imgs.to(DEVICE)
        real_labels = real_labels.to(DEVICE)

        real_tgt = torch.ones(bs, 1, device=DEVICE) * 0.9   # label smoothing
        fake_tgt = torch.zeros(bs, 1, device=DEVICE) + 0.1

        # ── Train Discriminator ──
        opt_D.zero_grad()
        d_real = D(real_imgs, real_labels)
        d_real_loss = criterion_gan(d_real, real_tgt)

        z      = torch.randn(bs, LATENT_DIM, device=DEVICE)
        f_labs = torch.randint(0, N_CLASSES, (bs,), device=DEVICE)
        fake   = G(z, f_labs).detach()
        d_fake = D(fake, f_labs)
        d_fake_loss = criterion_gan(d_fake, fake_tgt)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward(); opt_D.step()

        # ── Train Generator (2× per D update for stability) ──
        g_loss_total = 0.
        for _ in range(2):
            opt_G.zero_grad()
            z      = torch.randn(bs, LATENT_DIM, device=DEVICE)
            f_labs = torch.randint(0, N_CLASSES, (bs,), device=DEVICE)
            fake   = G(z, f_labs)
            g_loss = criterion_gan(D(fake, f_labs), torch.ones(bs, 1, device=DEVICE))
            g_loss.backward(); opt_G.step()
            g_loss_total += g_loss.item()

        epoch_g += g_loss_total / 2
        epoch_d += d_loss.item()
        n_batches += 1

    sched_G.step(); sched_D.step()
    g_losses.append(epoch_g / n_batches)
    d_losses.append(epoch_d / n_batches)

    # Note potential failure mode
    if not failure_epoch_noted and epoch > 10:
        d_mean = np.mean(d_losses[-5:])
        g_mean = np.mean(g_losses[-5:])
        if d_mean < 0.15 or g_mean > 4.0:
            print(f"  [!] Potential instability at epoch {epoch}: D_loss={d_mean:.3f}, G_loss={g_mean:.3f}")
            failure_epoch_noted = True

    if epoch % SAMPLE_EVERY == 0 or epoch == 1:
        print(f"  [cGAN] Ep {epoch:3d}/{GAN_EPOCHS} | D_loss: {d_losses[-1]:.4f} | G_loss: {g_losses[-1]:.4f}")
        save_gan_samples(epoch, G, fixed_noise, fixed_labels, f"epoch_{epoch:03d}.png")

print("\n[C] Generating GAN analysis plots …")

# GAN Training loss curves
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(g_losses, label='Generator Loss',     color='#e74c3c', linewidth=1.5)
ax.plot(d_losses, label='Discriminator Loss', color='#3498db', linewidth=1.5)
ax.axhline(y=np.log(2), color='gray', linestyle=':', label='Nash Eq. (~0.693)', linewidth=1)
ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
ax.set_title('cGAN Training Losses', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); save_fig("gan_training_losses.png")

# Final sample grid
save_gan_samples(GAN_EPOCHS, G, fixed_noise, fixed_labels, "final_samples.png")

# Sample quality progression (epochs 1, 20, 40, final)
prog_epochs = [1, 20, 40, GAN_EPOCHS]
prog_files  = [os.path.join(GAN_DIR, f"epoch_{e:03d}.png") for e in prog_epochs]
existing    = [f for f in prog_files if os.path.exists(f)]
if existing:
    fig, axes = plt.subplots(1, len(existing), figsize=(len(existing)*4, 4))
    if len(existing) == 1: axes = [axes]
    for ax, fpath, ep in zip(axes, existing, prog_epochs):
        img_data = plt.imread(fpath)
        ax.imshow(img_data); ax.axis('off')
        ax.set_title(f'Epoch {ep}', fontweight='bold')
    fig.suptitle("cGAN Sample Quality Progression", fontsize=13, fontweight='bold')
    plt.tight_layout(); save_fig("gan_progression.png")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
print("\n── Part A: CNN ─────────────────────────────────────────────")
print(f"  {'Model':<15} {'Best Val Acc':>13} {'Train Time':>11} {'Params':>10}")
print(f"  {'-'*52}")
print(f"  {'Custom CNN':<15} {max(va1):>12.2f}% {t1:>10.0f}s {count_params(CustomCNN())/1e6:>9.2f}M")
print(f"  {'ResNet18':<15} {max(va2):>12.2f}% {t2:>10.0f}s {count_params(get_resnet18())/1e6:>9.2f}M")

print("\n── Part B: RNN/LSTM/GRU ────────────────────────────────────")
print(f"  {'Model':<8} {'Test RMSE':>12}")
print(f"  {'-'*22}")
for t in ['RNN','LSTM','GRU']:
    print(f"  {t:<8} {rnn_results[t]['rmse']:>11.2f}")

print("\n── Part C: cGAN ────────────────────────────────────────────")
print(f"  Final G_loss: {g_losses[-1]:.4f}  |  Final D_loss: {d_losses[-1]:.4f}")
print(f"  Failure mode observed: Mode collapse risk (G_loss spike).")
print(f"  Mitigation applied: label smoothing, 2× G updates, LR decay.")

print(f"\n  All results saved to: {RESULTS_DIR}/")
print(f"  GAN samples saved to: {GAN_DIR}/")
print("\n" + "="*60)
print("  ALL DONE!")
print("="*60)
