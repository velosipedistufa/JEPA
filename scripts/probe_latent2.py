#!/usr/bin/env python3
"""Probe: run one train forward pass, compute latent_std."""
import json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "le-wm"))

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from functools import partial

import stable_worldmodel as swm
import stable_pretraining as spt
from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor


@hydra.main(version_base=None, config_path=str(ROOT / "le-wm/config/train"), config_name="lewm")
def probe(cfg):
    # Dataset
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))

    # Model
    world_model = hydra.utils.instantiate(cfg.model)
    world_model.eval()

    with torch.no_grad():
        output = world_model.encode(batch)
        emb = output["emb"]

    latent_std = emb.std().item()
    print(json.dumps({"latent_std": latent_std}))


if __name__ == "__main__":
    probe()