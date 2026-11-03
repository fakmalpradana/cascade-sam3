"""SAM 3 lewat HF Transformers: prompt kotak (Sam3Tracker) dan prompt teks (Sam3Model)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .instances import Instances

log = logging.getLogger(__name__)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_dtype(name: str):
    return getattr(torch, name)


def tiny_tracker_config():
    """Model mini untuk tes CPU tanpa unduh bobot (±6,8 jt parameter)."""
    from transformers import Sam3TrackerConfig

    c = Sam3TrackerConfig()
    b = c.vision_config.backbone_config
    b.hidden_size, b.intermediate_size, b.num_hidden_layers = 64, 128, 2
    b.num_attention_heads, b.global_attn_indexes = 2, [1]
    return c


def decoder_linear_names(model) -> list[str]:
    return [n for n, m in model.mask_decoder.named_modules() if isinstance(m, torch.nn.Linear)]


def apply_lora(model, r: int, alpha: int):
    from peft import LoraConfig, inject_adapter_in_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=decoder_linear_names(model))
    model.mask_decoder = inject_adapter_in_model(cfg, model.mask_decoder)
    return model


def load_decoder_weights(model, ckpt_path: str | Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("mode") == "lora":
        apply_lora(model, ck["lora_r"], ck["lora_alpha"])
    model.mask_decoder.load_state_dict(ck["state_dict"])
    log.info("bobot decoder dimuat: %s (epoch %s, val_iou %.4f)", ckpt_path, ck.get("epoch"), ck.get("val_iou", -1))
    return ck


def load_tracker(model_id: str, dtype: str = "bfloat16", device: str | None = None, decoder_ckpt=None):
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    model = Sam3TrackerModel.from_pretrained(model_id, dtype=torch.float32)
    processor = Sam3TrackerProcessor.from_pretrained(model_id)
    if decoder_ckpt:
        load_decoder_weights(model, decoder_ckpt)
    model.to(device or default_device(), dtype=torch_dtype(dtype)).eval()
    return model, processor


class BoxRefiner:
    """Embedding sekali per tile, kotak diproses per batch."""

    def __init__(self, model, processor, batch: int = 32):
        self.model, self.processor, self.batch = model, processor, batch
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, boxes: np.ndarray):
        inputs = self.processor(images=rgb, input_boxes=[boxes.tolist()], return_tensors="pt")
        emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device, self.dtype))
        masks, ious = [], []
        for i in range(0, len(boxes), self.batch):
            out = self.model(image_embeddings=emb,
                             input_boxes=inputs["input_boxes"][:, i:i + self.batch].to(self.device, self.dtype),
                             multimask_output=False)
            m = self.processor.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"])[0]
            masks.append(m[:, 0].numpy().astype(bool))
            ious.append(out.iou_scores[0, :, 0].float().cpu().numpy())
        return np.concatenate(masks), np.concatenate(ious)


class ConceptSegmenter:
    """SAM 3 prompt teks (PCS). Embedding teks di-cache per prompt."""

    def __init__(self, model_id: str, prompts, threshold: float, dtype: str = "bfloat16", device=None):
        from transformers import Sam3Model, Sam3Processor

        self.device = device or default_device()
        self.dtype = torch_dtype(dtype)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.threshold = threshold
        self.text = {}
        with torch.no_grad():
            for p in prompts:
                t = self.processor(text=p, return_tensors="pt").to(self.device)
                emb = self.model.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
                self.text[p] = (emb, t["attention_mask"])

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> Instances:
        h, w = rgb.shape[:2]
        img = self.processor(images=rgb, return_tensors="pt")
        vision = self.model.get_vision_features(pixel_values=img["pixel_values"].to(self.device, self.dtype))
        masks, scores = [], []
        for emb, att in self.text.values():
            out = self.model(vision_embeds=vision, text_embeds=emb, attention_mask=att)
            res = self.processor.post_process_instance_segmentation(
                out, threshold=self.threshold, mask_threshold=0.5, target_sizes=[(h, w)])[0]
            if len(res["scores"]):
                masks.append(res["masks"].cpu().numpy().astype(bool))
                scores.append(res["scores"].float().cpu().numpy())
        if not masks:
            return Instances.empty(h, w)
        s = np.concatenate(scores)
        return Instances(np.concatenate(masks), s, ["concept"] * len(s))
