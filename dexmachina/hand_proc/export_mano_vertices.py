"""Export ARCTIC MANO hand meshes (world frame, every frame of a sequence) and each vertex's dominant skinning joint, for the
hand-object contact targets (ho_contact_assets.py human). Needs smplx, so run it with the ARCTIC environment:

  python dexmachina/hand_proc/export_mano_vertices.py --raw <arctic_data/data/raw_seqs> --mano <body_models/mano> \
      --out <dir> s01/ketchup_use_02 s02/notebook_use_02
"""
import argparse
import os

import numpy as np
import torch
from smplx import MANO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, help='ARCTIC raw_seqs directory (<subject>/<sequence>.mano.npy)')
    parser.add_argument('--mano', required=True, help='directory with MANO_LEFT.pkl / MANO_RIGHT.pkl')
    parser.add_argument('--out', required=True)
    parser.add_argument('sequences', nargs='+', help='<subject>/<sequence>, e.g. s01/ketchup_use_02')
    args = parser.parse_args()
    layers = {s: MANO(args.mano, is_rhand=(s == 'right'), use_pca=False, flat_hand_mean=False, create_transl=True)
              for s in ('left', 'right')}
    for seq in args.sequences:
        subject, name = seq.split('/')
        raw = np.load(os.path.join(args.raw, subject, f'{name}.mano.npy'), allow_pickle=True).item()
        out = dict()
        for side, layer in layers.items():
            r = raw[side]
            frames = len(r['rot'])
            with torch.no_grad():
                o = layer(global_orient=torch.tensor(r['rot']), hand_pose=torch.tensor(r['pose']),
                          betas=torch.tensor(np.tile(r['shape'][None], (frames, 1))), transl=torch.tensor(r['trans']))
            out[f'verts_{side}'] = o.vertices.numpy().astype(np.float32)               # (T, 778, 3)
            out[f'joints_{side}'] = o.joints.numpy().astype(np.float32)                # (T, 16 + 5 tips, 3)
            out[f'vertex_joint_{side}'] = layer.lbs_weights.numpy().argmax(1).astype(np.int16)   # (778,) MANO 16-joint order
        os.makedirs(os.path.join(args.out, subject), exist_ok=True)
        np.savez_compressed(os.path.join(args.out, subject, f'{name}.npz'), **out)
        print('saved', seq, out['verts_left'].shape)


if __name__ == '__main__':
    main()
