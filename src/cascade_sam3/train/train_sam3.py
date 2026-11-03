"""Fine-tune mask decoder SAM 3 (atau LoRA) dengan prompt kotak dari COCO yang sama."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import work
from ..models.sam3 import apply_lora, default_device, torch_dtype

log = logging.getLogger(__name__)


def masks_from_anns(anns, h, w) -> np.ndarray:
    out = np.zeros((len(anns), h, w), np.uint8)
    for k, a in enumerate(anns):
        for seg in a["segmentation"]:
            cv2.fillPoly(out[k], [np.round(np.array(seg).reshape(-1, 2)).astype(np.int32)], 1)
    return out


def boxes_from_masks(m: np.ndarray) -> np.ndarray:
    b = np.zeros((len(m), 4), np.float32)
    for i, x in enumerate(m):
        r, c = np.flatnonzero(x.any(1)), np.flatnonzero(x.any(0))
        b[i] = (c[0], r[0], c[-1] + 1, r[-1] + 1) if len(r) else (0, 0, 1, 1)
    return b


class BoxPromptDataset(Dataset):
    def __init__(self, coco_dir, split, processor, max_instances=24, jitter=0.1, mask_size=512,
                 max_images=None, train=True, seed=42):
        d = Path(coco_dir) / split
        coco = json.loads((d / "_annotations.coco.json").read_text(encoding="utf-8"))
        by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        self.items = [(d / im["file_name"], by_img[im["id"]]) for im in coco["images"] if im["id"] in by_img]
        if max_images and len(self.items) > max_images:
            idx = np.random.default_rng(seed).choice(len(self.items), max_images, replace=False)
            self.items = [self.items[i] for i in sorted(idx)]
        self.processor, self.max_instances, self.mask_size = processor, max_instances, mask_size
        self.train, self.seed = train, seed
        self.jitter = jitter if train else 0.05

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, anns = self.items[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + i)
        rgb = cv2.imread(str(path))[..., ::-1]
        if len(anns) > self.max_instances:
            anns = [anns[k] for k in rng.choice(len(anns), self.max_instances, replace=False)]
        h, w = rgb.shape[:2]
        gt = masks_from_anns(anns, h, w)
        if self.train:  # augmentasi dihedral
            k = int(rng.integers(4))
            rgb, gt = np.rot90(rgb, k), np.rot90(gt, k, axes=(1, 2))
            if rng.random() < 0.5:
                rgb, gt = rgb[:, ::-1], gt[:, :, ::-1]
        rgb, gt = np.ascontiguousarray(rgb), np.ascontiguousarray(gt)
        keep = gt.any((1, 2))
        gt = gt[keep]
        h, w = rgb.shape[:2]
        boxes = boxes_from_masks(gt)
        wh = boxes[:, 2:] - boxes[:, :2]
        boxes = boxes + rng.normal(0, self.jitter, boxes.shape) * np.concatenate([wh, wh], 1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        inputs = self.processor(images=rgb, input_boxes=[boxes.tolist()], return_tensors="pt")
        s = self.mask_size
        gt_r = np.stack([cv2.resize(m.astype(np.float32), (s, s), interpolation=cv2.INTER_AREA) > 0.5 for m in gt])
        return inputs["pixel_values"][0], inputs["input_boxes"][0], torch.from_numpy(gt_r.astype(np.float32))


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    loss = ce * (1 - pt) ** gamma
    loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.mean()


def dice_loss(logits, target, eps=1.0):
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def mask_iou(logits, target):
    p = (logits > 0).float().flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return inter / ((p + t - p * t).sum(1)).clamp(min=1)


def forward(model, pixel_values, boxes, device, enc_dtype, size):
    amp = torch.autocast(device_type=device.split(":")[0], dtype=enc_dtype)
    with torch.no_grad(), amp:
        emb = model.get_image_embeddings(pixel_values[None].to(device, enc_dtype))
    emb = [e.float() for e in emb]
    with amp:
        out = model(image_embeddings=emb, input_boxes=boxes[None].to(device), multimask_output=False)
    logits = F.interpolate(out.pred_masks[0, :, 0].float()[:, None], size=(size, size), mode="bilinear",
                           align_corners=False)[:, 0]
    return logits, out.iou_scores[0, :, 0].float()


@torch.no_grad()
def evaluate(model, loader, device, enc_dtype, size) -> float:
    model.mask_decoder.eval()
    ious = []
    for pv, boxes, gt in loader:
        logits, _ = forward(model, pv, boxes, device, enc_dtype, size)
        ious.append(mask_iou(logits, gt.to(device)).cpu())
    model.mask_decoder.train()
    return float(torch.cat(ious).mean()) if ious else float("nan")


def _collate(batch):
    return batch[0]


def train_sam3(cfg: dict, name: str | None = None, resume: bool = False, epochs: int | None = None,
               model=None, processor=None, device: str | None = None) -> Path:
    sc = cfg["sam3"]
    device = device or default_device()
    enc_dtype = torch_dtype(sc["amp_dtype"])
    out = work(cfg, "runs", "sam3", *([name] if name else []))
    out.mkdir(parents=True, exist_ok=True)
    if model is None:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        model = Sam3TrackerModel.from_pretrained(sc["model_id"], dtype=torch.float32)
        processor = Sam3TrackerProcessor.from_pretrained(sc["model_id"])
    for p in model.parameters():
        p.requires_grad_(False)
    if sc["train_mode"] == "lora":
        apply_lora(model, sc["lora_r"], sc["lora_alpha"])
    elif sc["train_mode"] == "decoder":
        for p in model.mask_decoder.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"sam3.train_mode tidak dikenal: {sc['train_mode']}")
    model.vision_encoder.to(enc_dtype)
    model.to(device)
    model.mask_decoder.train()
    params = [p for p in model.parameters() if p.requires_grad]
    log.info("parameter dilatih: %.2f jt (%s)", sum(p.numel() for p in params) / 1e6, sc["train_mode"])

    coco = work(cfg, "dataset", "coco")
    common = dict(processor=processor, max_instances=sc["max_instances_per_image"], mask_size=sc["mask_loss_size"],
                  seed=cfg["data"]["seed"])
    tr = BoxPromptDataset(coco, "train", jitter=sc["box_jitter"], max_images=sc["max_train_images"], train=True,
                          **common)
    va = BoxPromptDataset(coco, "valid", max_images=sc["max_val_images"], train=False, **common)
    nw = sc["num_workers"]
    tl = DataLoader(tr, batch_size=1, shuffle=True, num_workers=nw, collate_fn=_collate, persistent_workers=nw > 0)
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=nw, collate_fn=_collate)

    n_epochs = epochs or sc["epochs"]
    total = max(1, n_epochs * len(tl))
    warm = min(500, max(1, total // 20))
    opt = torch.optim.AdamW(params, lr=sc["lr"], weight_decay=sc["weight_decay"])
    def lr_lambda(s):
        return (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    start, best = 0, -1.0
    last_path, best_path = out / "sam3_decoder_last.pt", out / "sam3_decoder_best.pt"
    if resume:
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.mask_decoder.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best = ck["epoch"] + 1, ck["best"]
        log.info("resume dari epoch %d (best %.4f)", start, best)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(out / "tb"))
    except Exception:  # noqa: BLE001 - tensorboard opsional
        pass
    log_path = out / "train_log.csv"
    new_log = not log_path.exists() or not resume
    logf = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    w = csv.writer(logf)
    if new_log:
        w.writerow(["epoch", "train_loss", "val_iou", "lr", "seconds", "peak_vram_gb"])
    size = sc["mask_loss_size"]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        if start == 0:
            z = evaluate(model, vl, device, enc_dtype, size)
            log.info("mIoU val zero-shot: %.4f", z)
            w.writerow([0, "", z, "", "", ""])
            logf.flush()
            if writer:
                writer.add_scalar("val/miou", z, 0)
        step = start * len(tl)
        for ep in range(start, n_epochs):
            t0, losses = time.time(), []
            for pv, boxes, gt in tl:
                gt = gt.to(device)
                logits, iou_pred = forward(model, pv, boxes, device, enc_dtype, size)
                logits = logits.float()
                loss = sc["focal_weight"] * focal_loss(logits, gt) + sc["dice_weight"] * dice_loss(logits, gt) \
                    + sc["iou_weight"] * F.mse_loss(iou_pred, mask_iou(logits.detach(), gt))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
                step += 1
                if writer and step % 20 == 0:
                    writer.add_scalar("train/loss", loss.item(), step)
            v = evaluate(model, vl, device, enc_dtype, size)
            vram = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
            dt = time.time() - t0
            log.info("epoch %d: loss %.4f, val mIoU %.4f, %.0f s, VRAM puncak %.2f GB", ep + 1, np.mean(losses), v,
                     dt, vram)
            w.writerow([ep + 1, float(np.mean(losses)), v, sched.get_last_lr()[0], round(dt, 1), round(vram, 2)])
            logf.flush()
            if writer:
                writer.add_scalar("val/miou", v, ep + 1)
            best = max(best, v)
            ck = dict(mode=sc["train_mode"], lora_r=sc["lora_r"], lora_alpha=sc["lora_alpha"],
                      state_dict=model.mask_decoder.state_dict(), optimizer=opt.state_dict(),
                      scheduler=sched.state_dict(), epoch=ep, val_iou=v, best=best, model_id=sc["model_id"])
            torch.save(ck, last_path)
            if v >= best:
                torch.save(ck, best_path)
    finally:
        logf.close()
        if writer:
            writer.close()
    return best_path
