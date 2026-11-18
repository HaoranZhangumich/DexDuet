"""Train DexDuet on one of the seven ARCTIC tasks of the DexMachina suite, then evaluate it.

DexDuet = the per-hand policy (one actor per hand with a message from the other hand; per-hand PPO ratio, value and
advantage), assisted reference-state initialization (A-RSI: start bins with their own object assistance, 25% of starts at
frame 0, keyframes kept only once their episode reaches the clip end), the hand-graph reward (weight 0.5) and the
hand-object contact reward (weight 1.0), on top of DexMachina's rewards. The flags below give the configuration of the
paper's runs; only the run name and the seed differ. ("Coordinates Hand" was the project's working name and survives in
this file name and in the default run names.)

  python scripts/train_coordinates_hand.py --task Ketchup-300 --dry-run         # print the commands
  CUDA_VISIBLE_DEVICES=0 python scripts/train_coordinates_hand.py --task Ketchup-300 --seed 15
  CUDA_VISIBLE_DEVICES=0 python scripts/train_coordinates_hand.py --task Box-200 --skip-train   # evaluate an existing run
  CUDA_VISIBLE_DEVICES=0 python scripts/train_coordinates_hand.py --task Box-200 --smoke        # 20 updates, 256 envs: setup check
  CUDA_VISIBLE_DEVICES=0 python scripts/train_coordinates_hand.py --task Box-200 --hand allegro_hand   # another hand

Other hands (--hand, default inspire_hand): the hand-graph and contact rewards follow the hand's finger correspondence
(dexmachina/envs/finger_spec.py), the fingertip-gap scale is the hand's retargeting scaling_factor unless --hand-graph-scale
is given, and the long clips use DexMachina's non-Inspire weights (imitation 0.2, contact 2.0, bc 0.2: the authors' released
configs of all 45 Allegro/XHand/Schunk runs on Ketchup-300, Notebook-300 and Waffleiron-300).
The hand needs its retargeted trajectories and contacts (examples/retarget_missing.sh, contact_retarget_missing.sh). The
Inspire commands are unchanged.

Run from the repository root with the dexmachina environment.

After training, the native best-training-reward checkpoint is evaluated with the paper's protocol (100 lanes, seed 1042,
deterministic, corrected first observation) and scored with ADD-AUC (dexmachina/eval/compute_add.py), then re-recorded
from the same start state and scored with Contact-AUC (dexmachina/eval/slip_record.py + slip_auc.py; needs the MANO export,
--mano or $CH_MANO_EXPORT).

The repository goes first on the children's PYTHONPATH; entries already on PYTHONPATH (for example a patched Genesis
checkout) follow it.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HANDS = ('inspire_hand', 'allegro_hand', 'xhand', 'schunk_hand', 'dex3_hand', 'ability_hand', 'dexrobot_hand')

# task: (clip, max_epochs, num_zero_epoch, lower gain ratio, imitation, contact, bc, wait_epochs, deque_len,
#        rsi_bin_frames (0 = 20 equal bins), rsi_deadline, rsi_zero_ramp, curr_rew_thres)
TASKS = {
    'Ketchup-300': ('ketchup-40-340-s01-u02', 13000, 1000, 0.5, 0.3, 3.0, 0.3, 100, 40, 0, 9000, 1000, '0.6 0 0 0'),
    'Waffleiron-300': ('waffleiron-40-340-s01-u01', 13000, 1000, 0.5, 0.3, 3.0, 0.3, 100, 40, 15, 9000, 1000, '0.6 0 0 0'),
    'Mixer-300': ('mixer-40-340-s01-u01', 10000, 1000, 0.6, 0.2, 2.0, 0.2, 200, 40, 15, 6920, 770, '0.6 0.01 0.01 0.01'),
    'Notebook-300': ('notebook-40-340-s02-u02', 13000, 1000, 0.5, 0.3, 3.0, 0.3, 100, 40, 0, 9000, 1000, '0.6 0 0 0'),
    'Ketchup-100': ('ketchup-30-130-s01-u01', 5000, 500, 0.6, 0.2, 2.0, 0.2, 200, 20, 15, 3460, 380, '0.6 0 0 0'),
    'Mixer-170': ('mixer-30-200-s01-u01', 5000, 500, 0.6, 0.2, 2.0, 0.2, 200, 20, 15, 3460, 380, '0.6 0 0 0'),
    'Box-200': ('box-30-230-s01-u01', 5000, 500, 0.6, 0.2, 2.0, 0.2, 200, 20, 15, 3460, 380, '0.6 0 0 0'),
}
# DexMachina's long-clip weights for the hands other than Inspire, from the authors' released configs (runs f422-low5-th0:
# imi 0.2, con 2.0, bc 0.2 for Allegro, XHand and Schunk on all 15 runs per hand); the other tasks use the same weights for
# every hand. (examples/reproduce_benchmark.sh keeps bc 0.3 there, which the released runs did not use.)
OTHER_HAND_LONG_CLIP = {'imitation': 0.2, 'contact': 2.0, 'bc': 0.2}
LONG_CLIP_TASKS = ('Ketchup-300', 'Waffleiron-300', 'Notebook-300')

COMMON = """-B 12000 -obf -obt --actuate_object --retarget_name para --horizon 16 -imw 0.5 --gain_mode all --curr_schedule rsi
--learning_rate 0.0003 --contact_beta 10 --upper_ratios 0.9 0.9 1 --save_freq 5000 --group_collisions --fixed_mode uniform
--uniform_mode slow --action_penalty 0.01 --dialback_ep_len 30 --skip_grad --task_rew_betas 10 1 5 --use_retarget_contact
--aux_reset_thres 0 0 0 -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 -ert 0.5 --hand HAND
--coordinate_actor --coord_ablation_id per-hand --per_hand_ppo --ablation_protocol dexmachina --value_mode per_hand --interval 5
--force_range_init 50 --grad_threshold 0.001 --dialback_min_epochs 500 --dialback_ratios 0.98 1 1 --resample_every_epoch -1
--study_profile pilot --rsi_uniform_mix 0.3 --rsi_credit_gamma 0.75 --rsi_fail_alpha 0.02 --rsi_success_up 0.8
--rsi_success_down 0.4 --rsi_min_count 64 --rsi_min_gap 40 --rsi_min_horizon 10 --rsi_keyframes --rsi_kf_capacity 256
--rsi_kf_prob 0.002 --rsi_kf_min_fill 32 --rsi_kf_min_task 0.5 --rsi_grace_bank 8 --rsi_kf_confirm 30 --rsi_kf_to_end
--rsi_frame0_share 0.25 --hand_graph_weight 0.5 --hand_graph_beta_dir 5 --hand_graph_beta_gap 20 --hand_graph_thumb 0.5
--hand_graph_scale SCALE --ho_contact_weight 1.0""".split()


def retarget_scale(hand):
    """The hand's retargeting scaling_factor (dexmachina/assets/<hand>/retarget_config.yaml), as the CLI writes it."""
    import yaml
    with open(REPO / 'dexmachina' / 'assets' / hand / 'retarget_config.yaml') as f:
        return float(yaml.safe_load(f)['scaling_factor'])


def hand_prefix(hand):
    """Run folders are logs/rl_games/<hand>/<prefix>-<experiment>_... (dexmachina/rl/train_rl_games.py)."""
    return str(hand).split('_')[0]


def training_command(task, seed, experiment, project, python, *, smoke=False, hand='inspire_hand', hand_graph_scale=None,
                     bc_weight=None):
    clip, epochs, zero, low, imi, con, bc, wait, deque, bin_frames, deadline, ramp, thres = TASKS[task]
    if hand != 'inspire_hand' and task in LONG_CLIP_TASKS:
        imi, con, bc = OTHER_HAND_LONG_CLIP['imitation'], OTHER_HAND_LONG_CLIP['contact'], OTHER_HAND_LONG_CLIP['bc']
    if bc_weight is not None:   # e.g. 0.3 on the long clips for the paper's Schunk runs
        bc = bc_weight
    scale = retarget_scale(hand) if hand_graph_scale is None else float(hand_graph_scale)
    per_task = ['--clip', clip, '--max_epochs', str(epochs), '--num_zero_epoch', str(zero),
                '--lower_ratios', str(low), str(low), '1', '-imi', str(imi), '-con', str(con), '-bc', str(bc),
                '--wait_epochs', str(wait), '--deque_len', str(deque), '--curr_rew_thres', *thres.split(),
                '--rsi_deadline', str(deadline), '--rsi_zero_ramp', str(ramp),
                *(['--rsi_bin_frames', str(bin_frames)] if bin_frames else ['--rsi_bins', '20'])]
    common = list(COMMON)
    common[common.index('HAND')] = hand
    common[common.index('SCALE')] = str(scale)
    command = [python, str(REPO / 'dexmachina/rl/train_rl_games.py'), *common, *per_task, '--seed', str(seed),
               '-exp', experiment, '--wandb_project', project]
    if smoke:   # a few minutes end to end: same flags, 256 envs, 20 updates, assistance off at the end
        for flag, value in (('-B', '256'), ('--max_epochs', '20'), ('--num_zero_epoch', '0'), ('--save_freq', '10'),
                            ('--study_profile', 'validation')):
            command[command.index(flag) + 1] = value
    return command


def run_dir(experiment, hand='inspire_hand'):
    runs = sorted((REPO / 'logs/rl_games' / hand).glob(f'{hand_prefix(hand)}-{experiment}_*'))
    if len(runs) != 1:
        sys.exit(f'expected one run folder for {experiment}, found {[str(r) for r in runs]}')
    return runs[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--hand', default='inspire_hand', choices=HANDS, help='robot hand (both sides)')
    ap.add_argument('--hand-graph-scale', type=float, default=None,
                    help="fingertip-gap scale of the hand graph (default: the hand's retargeting scaling_factor)")
    ap.add_argument('--bc-weight', type=float, default=None,
                    help="behavior-cloning weight instead of the task's default; the paper's Schunk runs use 0.3 on "
                         "Ketchup-300, Notebook-300 and Waffleiron-300 (DexMachina's Schunk release: 0.2)")
    ap.add_argument('--experiment', default=None, help='run name (default coordinates-hand_<task>_seed<seed>)')
    ap.add_argument('--project', default=os.environ.get('WANDB_PROJECT', 'coordinates-hand'))
    ap.add_argument('--reference', default=None, help=argparse.SUPPRESS)   # older option, no longer needed; ignored
    ap.add_argument('--reset-manifest', default=None,
                    help='evaluation start manifest (evaluation_reset.json); default resets/<task>/evaluation_reset.json, '
                         'created by the first evaluation and checked byte for byte afterwards')
    ap.add_argument('--mano', default=os.environ.get('CH_MANO_EXPORT', str(Path.home() / 'data/coordinates_hand/mano_export')),
                    help='MANO export (export_mano_vertices.py) for Contact-AUC')
    ap.add_argument('--skip-train', action='store_true', help='evaluate an existing run with this name')
    ap.add_argument('--smoke', action='store_true', help='setup check: 256 envs, 20 updates (run name gets _smoke)')
    ap.add_argument('--dry-run', action='store_true', help='print the commands and exit')
    args = ap.parse_args()

    python = sys.executable
    clip = TASKS[args.task][0]
    experiment = args.experiment or f'coordinates-hand_{args.task.lower()}_seed{args.seed}' + ('_smoke' if args.smoke else '')
    hand_tag = '' if args.hand == 'inspire_hand' else f'_{args.hand}'      # Inspire paths are unchanged
    manifest = Path(args.reset_manifest) if args.reset_manifest else (
        REPO / 'resets' / (args.task.lower() + hand_tag + ('_smoke' if args.smoke else '')) / 'evaluation_reset.json')
    pythonpath = os.pathsep.join([str(REPO)] + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    train = training_command(args.task, args.seed, experiment, args.project, python, smoke=args.smoke, hand=args.hand,
                             hand_graph_scale=args.hand_graph_scale, bc_weight=args.bc_weight)
    if args.dry_run:
        print(shlex.join(train))
        return
    if not args.skip_train:
        # the training subprocesses import Genesis through this interpreter's install: refuse the unpatched broad phase
        check = subprocess.run([python, '-c', 'from genesis.engine.solvers.rigid.collider_decomp import Collider; '
                                'import sys; sys.exit(0 if hasattr(Collider, "check_overflow") else 1)'],
                               cwd=REPO, env=env, capture_output=True)
        if check.returncode != 0:
            sys.exit('Genesis lacks the broad-phase overflow fix (fca4168, docs/METHOD.md, "Genesis collision-pair overflow"): '
                     'use an environment set up by scripts/setup_dexmachina_env.sh')
        subprocess.run(train, cwd=REPO, env=env, check=True)
    rd = run_dir(experiment, args.hand)
    checkpoint = rd / 'nn' / f'{args.hand}.pth'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python, '-m', 'dexmachina.rl.eval_rl_games', '--checkpoint', str(checkpoint),
                    '--num_envs', '100', '--all_policy_envs', '--eval-seed', '1042', '--eval-reset-manifest', str(manifest)],
                   cwd=REPO, env=env, check=True)
    subprocess.run([python, '-m', 'dexmachina.eval.compute_add', '--input',
                    str(rd / f'{args.hand}_eval_all_policy' / 'eval_ep0.npy'), '--obj_name', clip.split('-')[0]],
                   cwd=REPO, env=env, check=True)
    slip = rd / 'slip'
    if not (slip / 'recording.npz').exists():
        subprocess.run([python, '-m', 'dexmachina.eval.slip_record', '--checkpoint', str(checkpoint),
                        '--clip', clip, '--manifest', str(manifest), '--task', args.task, '--out', str(slip)],
                       cwd=REPO, env=env, check=True)
    subprocess.run([python, '-m', 'dexmachina.eval.slip_auc', str(slip), '--mano', args.mano], cwd=REPO, env=env, check=True)


if __name__ == '__main__':
    main()
