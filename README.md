<div align="center">

<h1>DexDuet</h1>

<h3>Learning Contact-Preserving Bimanual Dexterity from Human Demonstrations</h3>

<a href="https://github.com/RCL-Robotics/nv-v2d-hoi"><img alt="2nd place, V2D Challenge Track 2" src="https://img.shields.io/badge/V2D%20Challenge%20Track%202-2nd%20place-2F6B5E?style=flat-square"></a>
<img alt="Paper coming soon" src="https://img.shields.io/badge/Paper-coming%20soon-B7C4BF?style=flat-square">
<img alt="Project page coming soon" src="https://img.shields.io/badge/Project%20page-coming%20soon-B7C4BF?style=flat-square">
<a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-B7C4BF?style=flat-square"></a>

<br><br>

<img src="media/teaser.jpg" width="100%" alt="DexDuet preserves the human's contacts, with different robot hands and objects">

</div>

<br>

DexDuet learns a bimanual policy from one human demonstration and keeps the human's hand–object contacts. Our method
placed **2nd in Track 2 of the V2D Challenge**; the implementation for all three tiers is in
[RCL-Robotics/nv-v2d-hoi](https://github.com/RCL-Robotics/nv-v2d-hoi).

## Across robot hands

<p align="center"><img src="media/four_hands.gif" width="100%" alt="DexDuet on Mixer-300 with Inspire, Allegro, XHand and Schunk hands"></p>
<p align="center"><sub>Inspire · Allegro · XHand · Schunk</sub></p>

<p align="center"><img src="media/gallery.jpg" width="100%" alt="DexDuet with four robot hands on the seven DexMachina-suite demonstrations"></p>

## Compared with DexMachina

<table>
  <tr>
    <td align="center"><img src="media/box-200.gif" width="400" alt="Box-200"><br><sub>Box-200</sub></td>
    <td align="center"><img src="media/ketchup-100.gif" width="400" alt="Ketchup-100"><br><sub>Ketchup-100</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="media/ketchup-300.gif" width="400" alt="Ketchup-300"><br><sub>Ketchup-300</sub></td>
    <td align="center"><img src="media/waffleiron-300.gif" width="400" alt="Waffleiron-300"><br><sub>Waffleiron-300</sub></td>
  </tr>
</table>
<p align="center"><sub>DexMachina (left), DexDuet (right), with Inspire hands.</sub></p>

## Method

<p align="center"><img src="media/method.jpg" width="100%" alt="DexDuet overview"></p>

## Results

| Hand | ADD-AUC ↑ | Contact-AUC ↑ |
|:--|:-:|:-:|
| Inspire | 52.5 → **88.7** | 33.6 → **79.6** |
| Allegro | 65.8 → **94.8** | 35.8 → **68.0** |
| XHand | 60.7 → **93.6** | 38.2 → **79.5** |
| Schunk | 73.7 → **94.1** | 45.2 → **80.4** |

<sub>DexMachina → DexDuet on the DexMachina suite, 7 demonstrations × 5 seeds. The paper also reports the SPIDER suite, the ManipTrans suite and the V2D challenge.</sub>

## Get started

```bash
bash scripts/reproduce.sh setup                                # environments, patched Genesis
bash scripts/reproduce.sh download-arctic                      # ARCTIC, MANO, SMPL-X (your accounts)
bash scripts/reproduce.sh prepare                              # retarget the demonstrations
conda activate dexmachina

CUDA_VISIBLE_DEVICES=0 python scripts/train_coordinates_hand.py --task Box-200   # train and evaluate
bash scripts/reproduce_paper.sh from-release --suite main      # main table from our checkpoints
```

[Checkpoints](https://huggingface.co/oldTOM/dexduet-checkpoints) · [Reproduce the DexMachina-suite tables](REPRODUCE.md) · [Flags and layout](docs/DETAILS.md) · [G1 + Dex3 study](https://github.com/HaoranZhangumich/DexDuet-V2D) · [V2D Challenge, all three tiers](https://github.com/RCL-Robotics/nv-v2d-hoi)

## Acknowledgement

Built on [DexMachina](https://github.com/MandiZhao/dexmachina); thanks to its authors for releasing their code, assets
and checkpoints. MIT license, inherited from DexMachina. The ARCTIC object meshes in `dexmachina/assets/arctic/` and the
MANO hand assets in `dexmachina/assets/mano_hand/` come from DexMachina's repository; the terms of
[ARCTIC](https://arctic.is.tue.mpg.de/) and [MANO](https://mano.is.tue.mpg.de/) apply to them.

<details>
<summary>Citation</summary>

The DexDuet citation will follow on publication. Please also cite DexMachina:

```bibtex
@inproceedings{mandi2026dexmachina,
  author    = {Zhao Mandi and Yifan Hou and Dieter Fox and Yashraj Narang and Ajay Mandlekar and Shuran Song},
  title     = {{DexMachina}: Functional Retargeting for Bimanual Dexterous Manipulation},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

</details>
