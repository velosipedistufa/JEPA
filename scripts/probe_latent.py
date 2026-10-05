#!/usr/bin/env python3
"""Probe: load LeWM encoder, run one batch, print latent_std."""
import json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "le-wm"))

import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf

LEWM = Path(__file__).resolve().parents[1] / "le-wm"
cfg = OmegaConf.load(LEWM / "config/train/lewm.yaml")
model_cfg = OmegaConf.load(LEWM / "config/train/model/lewm.yaml")

# Load dataset to get action dim
ds_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
name = ds_cfg.pop("name")
dataset = swm.data.load_dataset(name, transform=None, **ds_cfg)

# Build model
from jepa import JEPA
model_cfg.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
model = JEPA(**model_cfg)
model.eval()

# Data transform
from stable_pretraining.data.transforms import Compose
from utils import get_img_preprocessor, get_column_normalizer
transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
for col in cfg.data.dataset.keys_to_load:
    if not col.startswith("pixels"):
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)
dataset.transform = Compose(*transforms)

loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))

with torch.no_grad():
    out = model.encode(batch)

latent_std = out["emb"].std().item()
print(json.dumps({"latent_std": latent_std}))