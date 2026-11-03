"""Image model: a 2-D ImageNet backbone on each of N slices (first conv collapsed to one channel),
attention pooling over the slices, a small head. Trained with differential learning rates
(backbone x0.05, pooling x0.5, head x1), OneCycle schedule, AMP, early stopping on validation
macro-F1; afterwards a temperature is fitted on the validation split so that the softmax
confidence is calibrated."""
from __future__ import annotations

import logging
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .volumes import VolumeLoader, augment

log = logging.getLogger(__name__)


def _one_channel(conv: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding,
                    bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _backbone(name: str, pretrained: bool = True):
    if name == "convnext_tiny":
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        net.features[0][0] = _one_channel(net.features[0][0])
        return net.features, 768
    if name == "efficientnet_b2":
        net = tvm.efficientnet_b2(weights=tvm.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        net.features[0][0] = _one_channel(net.features[0][0])
        return net.features, 1408
    raise ValueError(f"image_model.backbone must be convnext_tiny or efficientnet_b2, got {name!r}")


class SliceAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):                                   # (B, N, dim)
        return (x * torch.softmax(self.attn(x), dim=1)).sum(dim=1)


class ImageModel(nn.Module):
    def __init__(self, n_classes: int, backbone: str = "convnext_tiny", dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, self.feat_dim = _backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.slice_pool = SliceAttentionPool(self.feat_dim)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, n_classes))
        self.temperature = 1.0

    def forward(self, vols):                                # (B, N, H, W)
        b, n, h, w = vols.shape
        feats = self.pool(self.backbone(vols.view(b * n, 1, h, w))).view(b, n, self.feat_dim)
        return self.head(self.slice_pool(feats))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, weight=None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma, self.weight, self.ls = gamma, weight, label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits.float(), targets, weight=self.weight, label_smoothing=self.ls, reduction="none")
        return ((1 - torch.exp(-ce.detach())) ** self.gamma * ce).mean()


class _Volumes(Dataset):
    """(volume, label index) of rows; a volume that cannot be loaded is replaced by the next
    loadable one of the same split (at most 10 tries)."""

    def __init__(self, keys, labels, loader: VolumeLoader, training: bool):
        self.keys, self.labels, self.loader, self.training = list(keys), list(labels), loader, training

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        for k in range(min(11, len(self.keys))):
            j = (i + k) % len(self.keys)
            vol = self.loader(self.keys[j])
            if vol is not None:
                vol = augment(vol) if self.training else vol
                return torch.tensor(vol), torch.tensor(self.labels[j], dtype=torch.long)
        raise RuntimeError("no loadable volume among 11 consecutive samples of this split")


class _Keys(Dataset):
    def __init__(self, keys, loader: VolumeLoader):
        self.keys, self.loader = list(keys), loader

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        vol = self.loader(self.keys[i])
        ok = vol is not None
        if not ok:
            vol = np.zeros((self.loader.n_slices, self.loader.size, self.loader.size), np.float32)
        return torch.tensor(vol), ok


@torch.no_grad()
def _logits(model, loader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    out, ys = [], []
    for vols, y in loader:
        vols = torch.nan_to_num(vols.cuda(non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.amp.autocast("cuda"):
            out.append(model(vols).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """T minimising the NLL of softmax(logits / T) on the validation split (Guo et al. 2017)."""
    ok = torch.isfinite(logits).all(dim=1)
    logits, labels = logits[ok], labels[ok]
    if len(labels) < 2:
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def nll():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(nll)
    t = float(log_t.detach().exp())
    return t if 0.05 < t < 20 else 1.0


def train(rows, val_rows, classes: list[str], task: dict, cfg: dict, loader: VolumeLoader, out_path: str) -> dict:
    """Train on `rows`, select and calibrate on `val_rows` (DataFrames with key, label, raw_label).
    Saves the best checkpoint (with temperature) to out_path; returns {best_f1, temperature, epochs}."""
    mc = cfg["image_model"]
    cidx = {c: i for i, c in enumerate(classes)}
    y_tr = [cidx[l] for l in rows["label"]]
    y_va = [cidx[l] for l in val_rows["label"]]
    kw = dict(batch_size=mc["batch_size"], num_workers=mc["workers"], pin_memory=True,
              persistent_workers=mc["workers"] > 0)
    if task["balance_sampler"]:     # oversample rare raw labels (e.g. SPINE / OTHER inside REST)
        cnt = Counter(rows["raw_label"])
        w = torch.tensor([len(rows) / cnt[r] for r in rows["raw_label"]], dtype=torch.float32)
        tr_dl = DataLoader(_Volumes(rows["key"], y_tr, loader, True),
                           sampler=WeightedRandomSampler(w, len(rows), replacement=True), **kw)
    else:
        tr_dl = DataLoader(_Volumes(rows["key"], y_tr, loader, True), shuffle=True, **kw)
    va_dl = DataLoader(_Volumes(val_rows["key"], y_va, loader, False), shuffle=False, **kw)

    model = ImageModel(len(classes), mc["backbone"]).cuda()
    cnt = Counter(y_tr)
    weight = torch.tensor([min(len(y_tr) / (len(classes) * cnt.get(c, 1)), task["class_weight_cap"])
                           for c in range(len(classes))], dtype=torch.float32).cuda()
    loss_cfg = task["loss"]
    criterion = (FocalLoss(loss_cfg["gamma"], weight, loss_cfg["label_smoothing"]) if loss_cfg["type"] == "focal"
                 else nn.CrossEntropyLoss(weight=weight, label_smoothing=loss_cfg["label_smoothing"]))
    lr = mc["lr"]
    groups = [{"params": list(model.backbone.parameters()), "lr": lr * 0.05},
              {"params": list(model.slice_pool.parameters()), "lr": lr * 0.5},
              {"params": list(model.head.parameters()), "lr": lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                steps_per_epoch=len(tr_dl), epochs=mc["epochs"], pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    best, no_imp, t0, ep = -1.0, 0, time.time(), 0
    for ep in range(1, mc["epochs"] + 1):
        model.train()
        losses, skipped = [], 0
        for vols, y in tr_dl:
            vols = torch.nan_to_num(vols.cuda(non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(vols), y)
            if not loss.isfinite():
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prev = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= prev:      # no step on AMP-skipped batches: keeps OneCycle aligned
                sched.step()
            losses.append(loss.item())
        if not losses:
            raise RuntimeError(f"training diverged in epoch {ep}: every batch had a non-finite loss")
        logits, ys = _logits(model, va_dl)
        f1 = f1_score(ys.numpy(), torch.nan_to_num(logits).argmax(1).numpy(), average="macro", zero_division=0)
        log.info("epoch %d: train loss %.4f, val macro-F1 %.3f%s", ep, np.mean(losses), f1,
                 f" ({skipped} batches skipped)" if skipped else "")
        if f1 > best + 1e-4:
            best, no_imp = f1, 0
            save(model, classes, out_path)
        else:
            no_imp += 1
            if no_imp >= mc["patience"]:
                log.info("early stop")
                break
    if not os.path.exists(out_path) or os.path.getmtime(out_path) < t0:
        raise RuntimeError("no checkpoint was saved in this run")
    model, _ = load(out_path)
    model.temperature = fit_temperature(*_logits(model.cuda(), va_dl))
    save(model, classes, out_path)
    return {"best_val_f1": float(best), "temperature": model.temperature, "epochs": ep}


def save(model: ImageModel, classes: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "backbone": model.backbone_name,
                "temperature": model.temperature}, path)


def load(path: str) -> tuple[ImageModel, list[str]]:
    ck = torch.load(path, map_location="cpu")
    model = ImageModel(len(ck["classes"]), ck["backbone"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.temperature = float(ck.get("temperature", 1.0))
    return model, ck["classes"]


@torch.no_grad()
def predict(path: str, keys: list[str], loader: VolumeLoader, batch_size: int = 16, workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated probabilities (len(keys), n_classes) in the checkpoint's class order, and a mask of
    the keys whose volume could be loaded (rows of the others are NaN)."""
    model, classes = load(path)
    model = model.cuda().eval()
    dl = DataLoader(_Keys(keys, loader), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    probs, oks = [], []
    for vols, ok in dl:
        vols = torch.nan_to_num(vols.cuda(non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.amp.autocast("cuda"):
            logits = torch.nan_to_num(model(vols).float(), nan=0.0, posinf=0.0, neginf=0.0)
        probs.append(torch.softmax(logits / model.temperature, dim=1).cpu().numpy())
        oks.append(ok.numpy())
    p, ok = np.concatenate(probs), np.concatenate(oks).astype(bool)
    p[~ok] = np.nan
    return p, ok
