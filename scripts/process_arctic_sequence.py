#!/usr/bin/env python3
"""Process one ARCTIC sequence, with a CPU fallback for unsupported GPUs."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mano_p", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Import after parsing so `--help` works before licensed data is present;
    # ARCTIC's processing module reads metadata during import.
    from common.body_models import construct_layers
    from common.object_tensors import ObjectTensors
    from src.arctic.processing import process_seq

    with open("./data/arctic_data/data/meta/misc.json", encoding="utf-8") as stream:
        misc = json.load(stream)
    statcams = {
        subject: {
            "world2cam": torch.tensor(np.asarray(cfg["world2cam"]), dtype=torch.float32),
            "intris_mat": torch.tensor(np.asarray(cfg["intris_mat"]), dtype=torch.float32),
        }
        for subject, cfg in misc.items()
    }
    layers = construct_layers(args.device)
    object_tensor = ObjectTensors()
    object_tensor.to(args.device)
    layers["object"] = object_tensor

    progress = tqdm([args.mano_p])
    for mano_path in progress:
        progress.set_description(f"Processing {mano_path}")
        process_seq(
            [mano_path, args.device, statcams, layers, progress],
            export_verts=True,
        )


if __name__ == "__main__":
    main()
