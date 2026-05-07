import os
import csv
import cv2
import math
import glob
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# ============================================================
# 0. GLOBAL SPEED SETTINGS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Using device: {device}")

# ============================================================
# 1. CONFIG
# ============================================================
NUM_SAMPLES = 87000

CONFIG = {
    "bw_image_dir": "/kaggle/input/datasets/ulaganathankb/dataset/crops_bw",
    "csv_dir": "/kaggle/input/datasets/ulaganathankb/dataset/time_series",
    "output_dir": "/kaggle/working/ecg_training_out",

    "epochs": 100,
    "batch_size": 128,

    # Requested LR setup
    "max_lr": 3e-3,
    "min_lr_factor": 0.05,   # eta_min = max_lr * 0.05
    "warmup_epochs": 3,

    "weight_decay": 1e-4,
    "image_height": 256,
    "image_width": 250,
    "time_steps": 250,
    "patience": 10,

    "num_workers": min(os.cpu_count() or 2, 8),
    "alpha_gradient_weight": 5.0,
    "cosine_weight": 1.0,
    "xcorr_weight": 0.5,
    "max_grad_norm": 1.0,

    # Resume / checkpoint behavior
    "resume_from_best": False,
}

OUT_DIR = Path(CONFIG["output_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "loss_history.csv"
BEST_CKPT_PATH = OUT_DIR / "best_weight.pth"
LATEST_CKPT_PATH = OUT_DIR / "latest_checkpoint.pth"
TOTAL_LOSS_PNG = OUT_DIR / "total_loss.png"
HUBER_LOSS_PNG = OUT_DIR / "huber_loss.png"
COS_LOSS_PNG = OUT_DIR / "cos_loss.png"
XCORR_LOSS_PNG = OUT_DIR / "xcorr_loss.png"

# ============================================================
# 2. LEADS / WINDOWS
# ============================================================
LEAD_NAMES = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_TO_IDX = {name: i for i, name in enumerate(LEAD_NAMES)}
NUM_LEADS = len(LEAD_NAMES)

LEAD_WINDOWS = {
    'I': 0, 'II': 0, 'III': 0,
    'AVR': 1, 'AVL': 1, 'AVF': 1,
    'V1': 2, 'V2': 2, 'V3': 2,
    'V4': 3, 'V5': 3, 'V6': 3
}

# ============================================================
# 3. HELPERS
# ============================================================
#_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))


def get_state_dict(model):
    """Works for normal and torch.compile-wrapped models."""
    return model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def _load_signal(csv_path, lead_name, T):
    df       = pd.read_csv(csv_path)
    col_name = next((c for c in df.columns if c.upper() == lead_name), df.columns[0])
    full_sig = df[col_name].values.astype(np.float32)
    seg_idx  = LEAD_WINDOWS[lead_name]
    pts      = len(full_sig) // 4
    signal   = full_sig[seg_idx * pts : seg_idx * pts + pts]
    if len(signal) > T:
        sig_tensor = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        sig_tensor = F.interpolate(sig_tensor, size=T, mode='linear', align_corners=False)
        signal     = sig_tensor.squeeze(0).squeeze(0).numpy() 
    elif len(signal) < T:
        signal = np.pad(signal, (0, T - len(signal)), mode='edge')
    """
    std = signal.std()
    if std > 1e-6:
        signal = (signal - signal.mean()) / std
    """
    signal = signal * 10.0
    return signal.astype(np.float32)


def _load_and_prepare_image(img_path, H, W):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    new_h = max(1, int(h * (W / w)))
    img = cv2.resize(img, (W, new_h), interpolation=cv2.INTER_AREA)

    if new_h < H:
        pad_top = (H - new_h) // 2
        pad_bottom = H - new_h - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, 0)
    elif new_h > H:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    #img = cv2.dilate(img, _DILATE_KERNEL)
    return img


def ecg_augment(img, sig):
    H, W = img.shape

    if random.random() < 0.5:
        shift = random.randint(-15, 15)
        img = np.roll(img, shift, axis=1)
        sig = np.roll(sig, shift)

    if random.random() < 0.3:
        scale = random.uniform(0.88, 1.12)
        new_h = max(1, int(H * scale))
        img_s = cv2.resize(img, (W, new_h), interpolation=cv2.INTER_AREA)
        if new_h < H:
            pad = H - new_h
            img = cv2.copyMakeBorder(img_s, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, 0)
        else:
            start = (new_h - H) // 2
            img = img_s[start:start + H]

    if random.random() < 0.5:
        mask = img > 128
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img_int = img.astype(np.int16)
        img_int[mask] += noise[mask]
        img = np.clip(img_int, 0, 255).astype(np.uint8)

    return img, sig


def pearson_per_sample(pred, target, eps=1e-6):
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    pred = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target = target / (target.norm(dim=1, keepdim=True) + eps)
    return (pred * target).sum(dim=1)


def pp_error_per_sample(pred, target):
    pred_rng = pred.max(dim=1).values - pred.min(dim=1).values
    tgt_rng = target.max(dim=1).values - target.min(dim=1).values
    return (pred_rng - tgt_rng).abs()


# ============================================================
# 4. DATASET
# ============================================================
class ECGGroundTruthDataset(Dataset):
    def __init__(self, image_dir, csv_dir, num_samples):
        H, W, T = CONFIG["image_height"], CONFIG["image_width"], CONFIG["time_steps"]

        all_paths = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
        random.shuffle(all_paths)

        images, signals, lead_indices = [], [], []
        print(f"\nCaching up to {num_samples} samples from {len(all_paths)} images...")

        for img_path in all_paths:
            if len(images) >= num_samples:
                break

            basename = os.path.basename(img_path)
            parts = basename.replace(".jpg", "").replace(".png", "").split("_")
            lead_name = parts[-1].upper()
            csv_path = os.path.join(csv_dir, "_".join(parts[:-1]) + ".csv")

            if not (os.path.exists(csv_path) and lead_name in LEAD_WINDOWS):
                continue

            signal = _load_signal(csv_path, lead_name, T)
            img = _load_and_prepare_image(img_path, H, W)
            if img is None:
                continue

            images.append(np.ascontiguousarray(img))
            signals.append(np.ascontiguousarray(signal))
            lead_indices.append(LEAD_TO_IDX[lead_name])

        actual = len(images)
        print(f"Cached {actual} / {num_samples} samples.")

        self.images = np.stack(images, axis=0).astype(np.uint8)
        self.signals = np.stack(signals, axis=0).astype(np.float32)
        self.lead_indices = np.array(lead_indices, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.signals[idx], self.lead_indices[idx]


class ECGSubset(Dataset):
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, sig, lead_idx = self.subset[idx]
        if self.augment:
            img, sig = ecg_augment(img, sig)

        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        sig_t = torch.from_numpy(sig)
        lead_t = torch.tensor(lead_idx, dtype=torch.long)
        return img_t, sig_t, lead_t


# ============================================================
# 5. MODEL
# ============================================================
class SoftCentroid(nn.Module):
    def __init__(self, height):
        super().__init__()
        self.register_buffer(
            "y_coords",
            torch.arange(height, dtype=torch.float32).view(1, 1, height, 1),
            persistent=False
        )

    def forward(self, x):
        B, _, H, W = x.shape
        y_coords = self.y_coords.to(device=x.device, dtype=torch.float32)
        col_sums = x.sum(dim=2, keepdim=True) + 1e-6
        centroid = (x * y_coords).sum(dim=2, keepdim=True) / col_sums
        centroid = centroid.squeeze(2).squeeze(1)
        return 1.0 - centroid / (H - 1)


class DilatedCNN1D(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 48, kernel_size=5, padding=4, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, out_ch, kernel_size=5, padding=8, dilation=4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNN2D(nn.Module):
    def __init__(self, out_ch=128):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32, (2, 1)),
            self._block(32, 64, (2, 1)),
            self._block(64, 128, (2, 1)),
            self._block(128, 128, (2, 1)),
            self._block(128, 128, (2, 1)),
            self._block(128, out_ch, (2, 1)),
            self._block(out_ch, out_ch, (4, 1)),
        )

    @staticmethod
    def _block(ic, oc, pool):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool, pool)
        )

    def forward(self, x):
        return self.features(x).squeeze(2)


class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class ImprovedECGExtractor(nn.Module):
    def __init__(self, image_height=256, num_leads=NUM_LEADS, lead_emb_dim=16):
        super().__init__()
        self.soft_centroid = SoftCentroid(image_height)
        self.dilated_cnn = DilatedCNN1D(out_ch=64)
        self.cnn2d = CNN2D(out_ch=128)

        self.lead_embedding = nn.Embedding(num_leads, lead_emb_dim)
        self.lead_proj = nn.Linear(lead_emb_dim, 192)

        self.lstm = nn.LSTM(192, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = TemporalAttention(256, num_heads=4)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 1)

    def forward(self, x, lead_idx):
        centroid = self.soft_centroid(x)
        c1 = self.dilated_cnn(centroid.unsqueeze(1))
        c2 = self.cnn2d(x)
        fused = torch.cat([c1, c2], dim=1).transpose(1, 2)

        lead_emb = self.lead_embedding(lead_idx)
        lead_feat = self.lead_proj(lead_emb).unsqueeze(1)
        fused = fused + lead_feat

        out, _ = self.lstm(fused)
        out = self.attn(out)
        return self.fc(self.drop(out)).squeeze(2)


# ============================================================
# 6. LOSS
# ============================================================
class HybridECGLoss(nn.Module):
    def __init__(self, alpha=20.0, cos_weight=1.0, xcorr_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.cos_weight = cos_weight
        self.xcorr_weight = xcorr_weight
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.huber = nn.HuberLoss(reduction="none", delta=1.0)

    def _xcorr_loss(self, pred, target):
        B, T = pred.shape

        p = pred - pred.mean(dim=1, keepdim=True)
        t = target - target.mean(dim=1, keepdim=True)

        p = p / (p.norm(dim=1, keepdim=True) + 1e-6)
        t = t / (t.norm(dim=1, keepdim=True) + 1e-6)

        xcorr_at_zero = (p * t).sum(dim=1)

        n = 2 * T
        P = torch.fft.rfft(p, n=n)
        T_ = torch.fft.rfft(t, n=n)
        xcorr_all = torch.fft.irfft(P * T_.conj(), n=n)

        max_xcorr = xcorr_all.max(dim=1).values
        return F.relu(max_xcorr - xcorr_at_zero).mean()

    def forward(self, pred, target):
        grad = torch.diff(target, dim=1, prepend=target[:, :1])
        grad_w = 1.0 + self.alpha * grad.abs()
        amp_w = 1.0 + torch.abs(target) * 5.0
        total_w = grad_w * amp_w

        huber_loss = (self.huber(pred, target) * total_w).mean()
        cos_loss = (1.0 - self.cosine(pred, target)).mean()
        xcorr_loss = self._xcorr_loss(pred, target)

        cos_weighted = self.cos_weight * cos_loss
        xcorr_weighted = self.xcorr_weight * xcorr_loss

        total = huber_loss + cos_weighted + xcorr_weighted
        return total, huber_loss.detach(), cos_weighted.detach(), xcorr_weighted.detach()


# ============================================================
# 7. VALIDATION
# ============================================================
@torch.inference_mode()
def run_validation(model, val_loader, criterion):
    model.eval()

    total_sum = 0.0
    huber_sum = 0.0
    cos_sum = 0.0
    xcorr_sum = 0.0
    pearson_vals = []
    pp_err_vals = []

    lead_pearson = defaultdict(list)

    for images, targets, lead_indices in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lead_indices = lead_indices.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            preds = model(images, lead_indices)
            total, huber_l, cos_l, xcorr_l = criterion(preds, targets)

        bs = images.size(0)
        total_sum += total.detach().item() * bs
        huber_sum += huber_l.item() * bs
        cos_sum += cos_l.item() * bs
        xcorr_sum += xcorr_l.item() * bs

        batch_pearson = pearson_per_sample(preds, targets).detach().cpu().numpy()
        batch_pp = pp_error_per_sample(preds, targets).detach().cpu().numpy()
        lead_np = lead_indices.detach().cpu().numpy()

        pearson_vals.extend(batch_pearson.tolist())
        pp_err_vals.extend(batch_pp.tolist())

        for i, li in enumerate(lead_np):
            lead_pearson[LEAD_NAMES[li]].append(float(batch_pearson[i]))

    val_size = len(val_loader.dataset)
    val_total = total_sum / val_size
    val_huber = huber_sum / val_size
    val_cos = cos_sum / val_size
    val_xcorr = xcorr_sum / val_size

    pearson_overall = float(np.mean(pearson_vals)) if pearson_vals else float("nan")
    pp_err = float(np.mean(pp_err_vals)) if pp_err_vals else float("nan")
    per_lead_pearson = {k: float(np.mean(v)) for k, v in lead_pearson.items() if len(v) > 0}

    return val_total, val_huber, val_cos, val_xcorr, pearson_overall, pp_err, per_lead_pearson


# ============================================================
# 8. CHECKPOINTING
# ============================================================
def save_checkpoint(path, model, optimizer, scheduler_plateau, cosine_scheduler, scaler,
                    epoch, best_val_loss, history, model_cfg):
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_name": model.__class__.__name__,
        "model_config": model_cfg,
        "model_architecture": str(unwrap_model(model)),
        "model_state_dict": get_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_plateau_state_dict": scheduler_plateau.state_dict(),
        "cosine_scheduler_state_dict": cosine_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "config": CONFIG,
        "lead_names": LEAD_NAMES,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, scheduler_plateau, cosine_scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler_plateau is not None and "scheduler_plateau_state_dict" in ckpt:
        scheduler_plateau.load_state_dict(ckpt["scheduler_plateau_state_dict"])
    if cosine_scheduler is not None and "cosine_scheduler_state_dict" in ckpt:
        cosine_scheduler.load_state_dict(ckpt["cosine_scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# ============================================================
# 9. PLOTTING
# ============================================================
def plot_train_val(df, train_col, val_col, title, ylabel, outpath):
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df[train_col], marker="o", label=f"Train {ylabel}")
    plt.plot(df["epoch"], df[val_col], marker="o", label=f"Val {ylabel}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ============================================================
# 10. SETUP
# ============================================================
dataset = ECGGroundTruthDataset(CONFIG["bw_image_dir"], CONFIG["csv_dir"], NUM_SAMPLES)
total = len(dataset)
train_size = int(0.8 * total)
val_size = total - train_size
print(f"\nSplit -> Train: {train_size} | Val: {val_size}")

g = torch.Generator().manual_seed(42)
raw_train, raw_val = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

train_dataset = ECGSubset(raw_train, augment=True)
val_dataset = ECGSubset(raw_val, augment=False)

loader_kwargs = dict(
    num_workers=CONFIG["num_workers"],
    pin_memory=(device.type == "cuda"),
    persistent_workers=CONFIG["num_workers"] > 0,
)
if CONFIG["num_workers"] > 0:
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    drop_last=False,
    **loader_kwargs
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    drop_last=False,
    **loader_kwargs
)

model_cfg = {
    "image_height": CONFIG["image_height"],
    "num_leads": NUM_LEADS,
    "lead_emb_dim": 16,
}

raw_model = ImprovedECGExtractor(**model_cfg).to(device)

if hasattr(torch, "compile"):
    try:
        raw_model = torch.compile(raw_model)
        print("torch.compile() active.")
    except Exception:
        pass

criterion = HybridECGLoss(
    alpha=CONFIG["alpha_gradient_weight"],
    cos_weight=CONFIG["cosine_weight"],
    xcorr_weight=CONFIG["xcorr_weight"]
)

optimizer = torch.optim.AdamW(
    raw_model.parameters(),
    lr=CONFIG["max_lr"],
    weight_decay=CONFIG["weight_decay"]
)

# Requested scheduler kept exactly
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# Cosine scheduler with eta_min = max_lr * 0.05
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(1, CONFIG["epochs"] - CONFIG["warmup_epochs"]),
    eta_min=CONFIG["max_lr"] * CONFIG["min_lr_factor"]
)

scaler = torch.amp.GradScaler(device=device.type, enabled=USE_AMP)

# ============================================================
# 11. RESUME FROM BEST CHECKPOINT
# ============================================================
start_epoch = 0
best_val_loss = float("inf")
history = []

if CONFIG["resume_from_best"] and BEST_CKPT_PATH.exists():
    print(f"Resuming from: {BEST_CKPT_PATH}")
    ckpt = load_checkpoint(BEST_CKPT_PATH, unwrap_model(raw_model), optimizer, scheduler, cosine_scheduler, scaler)
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    history = ckpt.get("history", [])
    print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

# ============================================================
# 12. CSV SETUP AT START
# ============================================================
csv_fields = [
    "epoch",
    "lr",
    "train_total", "train_huber", "train_cos", "train_xcorr",
    "val_total", "val_huber", "val_cos", "val_xcorr",
    "pearson", "pp_error",
]
csv_fields += [f"pearson_{n}" for n in LEAD_NAMES]

csv_file = open(CSV_PATH, "a", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
if csv_file.tell() == 0:
    csv_writer.writeheader()
    csv_file.flush()
    os.fsync(csv_file.fileno())

# ============================================================
# 13. TRAINING LOOP
# ============================================================
epochs_without_improvement = 0

print("\nStarting training...\n")

for epoch in range(start_epoch, CONFIG["epochs"]):
    # Warm-up for the first epoch only
    if epoch == 0:
        set_lr(optimizer, CONFIG["max_lr"] * CONFIG["min_lr_factor"])
    elif epoch == CONFIG["warmup_epochs"]:
        set_lr(optimizer, CONFIG["max_lr"])

    raw_model.train()

    train_total_sum = 0.0
    train_huber_sum = 0.0
    train_cos_sum = 0.0
    train_xcorr_sum = 0.0

    for images, targets, lead_indices in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lead_indices = lead_indices.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            preds = raw_model(images, lead_indices)
            total, huber_l, cos_l, xcorr_l = criterion(preds, targets)

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), CONFIG["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        train_total_sum += total.detach().item() * bs
        train_huber_sum += huber_l.item() * bs
        train_cos_sum += cos_l.item() * bs
        train_xcorr_sum += xcorr_l.item() * bs

    train_total = train_total_sum / train_size
    train_huber = train_huber_sum / train_size
    train_cos = train_cos_sum / train_size
    train_xcorr = train_xcorr_sum / train_size

    val_total, val_huber, val_cos, val_xcorr, pearson, pp_err, per_lead_pearson = run_validation(
        raw_model, val_loader, criterion
    )

    # Planned cosine schedule after warm-up
    if epoch >= CONFIG["warmup_epochs"]:
        cosine_scheduler.step()

    # Plateau scheduler kept as requested
    scheduler.step(val_total)

    lr_now = current_lr(optimizer)

    row = {
        "epoch": epoch + 1,
        "lr": lr_now,
        "train_total": train_total,
        "train_huber": train_huber,
        "train_cos": train_cos,
        "train_xcorr": train_xcorr,
        "val_total": val_total,
        "val_huber": val_huber,
        "val_cos": val_cos,
        "val_xcorr": val_xcorr,
        "pearson": pearson,
        "pp_error": pp_err,
    }
    for lead in LEAD_NAMES:
        row[f"pearson_{lead}"] = per_lead_pearson.get(lead, float("nan"))

    history.append(row)

    # Append immediately to CSV
    csv_writer.writerow(row)
    csv_file.flush()
    os.fsync(csv_file.fileno())

    lead_str = "  ".join(f"{n}:{row[f'pearson_{n}']:.3f}" for n in LEAD_NAMES)

    print(
        f"Epoch {epoch + 1:03d} | "
        f"Train Total:{train_total:.4f} "
        f"(H:{train_huber:.4f} C:{train_cos:.4f} X:{train_xcorr:.4f}) | "
        f"Val Total:{val_total:.4f} "
        f"(H:{val_huber:.4f} C:{val_cos:.4f} X:{val_xcorr:.4f}) | "
        f"Pearson:{pearson:.4f} | P2P Err:{pp_err:.4f} | LR:{lr_now:.6f}"
    )
    print(f"           Per-lead Pearson -> {lead_str}")

    # Latest checkpoint every epoch
    save_checkpoint(
        LATEST_CKPT_PATH,
        raw_model,
        optimizer,
        scheduler,
        cosine_scheduler,
        scaler,
        epoch,
        best_val_loss,
        history,
        model_cfg
    )

    # Best checkpoint on validation improvement
    if val_total < best_val_loss:
        best_val_loss = val_total
        epochs_without_improvement = 0

        save_checkpoint(
            BEST_CKPT_PATH,
            raw_model,
            optimizer,
            scheduler,
            cosine_scheduler,
            scaler,
            epoch,
            best_val_loss,
            history,
            model_cfg
        )
        print("  -> New best model saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= CONFIG["patience"]:
        print("\nEarly stopping triggered.")
        break

csv_file.close()

# ============================================================
# 14. FINAL CSV + PLOTS
# ============================================================
df_hist = pd.DataFrame(history)
df_hist.to_csv(CSV_PATH, index=False)

plot_train_val(
    df_hist,
    "train_total",
    "val_total",
    "Total Loss Over Training",
    "Total Loss",
    TOTAL_LOSS_PNG
)

plot_train_val(
    df_hist,
    "train_huber",
    "val_huber",
    "Huber Loss Over Training",
    "Huber Loss",
    HUBER_LOSS_PNG
)

plot_train_val(
    df_hist,
    "train_cos",
    "val_cos",
    "Cosine Loss Over Training",
    "Cosine Loss",
    COS_LOSS_PNG
)

plot_train_val(
    df_hist,
    "train_xcorr",
    "val_xcorr",
    "Cross-correlation Loss Over Training",
    "XCorr Loss",
    XCORR_LOSS_PNG
)

print(f"\nSaved CSV: {CSV_PATH}")
print(f"Saved best checkpoint: {BEST_CKPT_PATH}")
print(f"Saved latest checkpoint: {LATEST_CKPT_PATH}")
print(f"Saved plots in: {OUT_DIR}")
