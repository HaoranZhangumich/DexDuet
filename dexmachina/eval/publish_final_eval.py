"""Publish a completed policy's mesh ADD-AUC to its original W&B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--wandb-project", default="dexmachina")
    parser.add_argument("--stats-path", help="Optional explicitly selected evaluation statistics")
    parser.add_argument("--checkpoint", help="Evaluated checkpoint, recorded with published metrics")
    parser.add_argument("--metric-prefix", default="final_eval",
                        help="Use a distinct prefix to publish another evaluation of the same run")
    args = parser.parse_args()
    if args.num_trials < 1 or not args.metric_prefix.strip('/'):
        parser.error('Positive num-trials and nonempty metric-prefix required')

    run_dir = Path(args.run_dir).resolve()
    stats_path = Path(args.stats_path).resolve() if args.stats_path else run_dir / "inspire_hand_eval" / "add_stats.json"
    if not stats_path.is_relative_to(run_dir):
        raise ValueError('Evaluation statistics must belong to the selected run')
    checkpoint = Path(args.checkpoint).resolve(strict=True) if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_relative_to(run_dir):
        raise ValueError('Checkpoint must belong to the selected run')
    env_path = run_dir / "params" / "env.yaml"
    with stats_path.open(encoding="utf-8") as stream:
        stats = json.load(stream)
    prefix = args.metric_prefix.strip('/')
    metrics = {
        f"{prefix}/num_trials": int(args.num_trials),
        f"{prefix}/add_auc": float(stats["auc"]["overall"]),
        f"{prefix}/mean_add": float(stats["mean_add"]["overall"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError('Refusing to publish non-finite evaluation statistics')
    protocol_path = stats_path.parent / 'eval_protocol.json'
    protocol = None
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        if (protocol.get('status') != 'complete'
                or protocol.get('completed_episodes') != protocol.get('eval_episodes')
                or protocol.get('num_trials') != args.num_trials):
            raise ValueError('Evaluation protocol is incomplete or trial count differs')
        if checkpoint is None:
            raise ValueError('Explicit checkpoint required with evaluation protocol metadata')
        if (Path(protocol['checkpoint']).resolve() != checkpoint
                or protocol['checkpoint_sha256'] != file_sha256(checkpoint)):
            raise ValueError('Evaluated checkpoint does not match protocol provenance')
        if (protocol['protocol'] == 'all_policy_controlled_v1'
                and protocol['policy_controlled_trials'] != args.num_trials):
            raise ValueError('All-policy evaluation contains uncontrolled trials')
        metrics[f'{prefix}/policy_controlled_trials'] = protocol['policy_controlled_trials']
    with env_path.open(encoding="utf-8") as stream:
        saved_environment = yaml.unsafe_load(stream)
        metadata = saved_environment.get("wandb", {})
    study = saved_environment.get('ablation_study', {})
    study_protocol = study.get('protocol')
    if study_protocol in ('v2', 'dexmachina'):
        if protocol is None or protocol['protocol'] != 'all_policy_controlled_v1':
            raise ValueError('Ablation publication requires complete all-policy evaluation provenance')
        reset = protocol.get('common_reset', {})
        if (args.num_trials != 100 or protocol.get('study_protocol') != study_protocol
                or protocol.get('evaluation_seed') != 1042
                or protocol.get('object_state_sampling') != 'post_physics_pre_reset_v2'
                or reset.get('seed') != 1042
                or reset.get('verification') != 'exact_arrays_equal_before_first_policy_action'):
            raise ValueError('Ablation evaluation is missing the common seeded reset verification')
        for file_key, hash_key in (('manifest', 'manifest_sha256'), ('state_file', 'state_file_sha256')):
            if file_sha256(Path(reset[file_key])) != reset[hash_key]:
                raise ValueError('Ablation evaluation reset provenance file has changed')
        if not (run_dir / '.training_complete').is_file():
            raise ValueError('Ablation training has not passed launcher checkpoint validation')
        if study_protocol == 'v2':
            endpoint = json.loads((run_dir / 'final_checkpoint.json').read_text())
            if (Path(endpoint['checkpoint']).resolve() != checkpoint
                    or endpoint.get('selection') != 'training_endpoint'):
                raise ValueError('v2 evaluation must use the declared training endpoint checkpoint')
            if study.get('profile') == 'pilot' and (
                endpoint.get('epoch') != 13000 or endpoint.get('environment_transitions') != 4992000000
            ):
                raise ValueError('v2 full-run budget does not match the study endpoint')
        else:
            from dexmachina.rl.final_checkpoint import validate_training_checkpoint
            with (run_dir / 'params' / 'agent.yaml').open() as stream:
                config = yaml.full_load(stream)['params']['config']
            selected_checkpoint, endpoint = validate_training_checkpoint(run_dir, config, study_protocol)
            if selected_checkpoint.resolve() != checkpoint:
                raise ValueError('DexMachina evaluation must use the native best training reward checkpoint')
            if (config.get('ablation_protocol') != 'dexmachina'
                    or config.get('lr_schedule') != 'adaptive'
                    or config.get('schedule_type') != 'standard'
                    or config.get('fixed_epoch_budget') is not False):
                raise ValueError('DexMachina saved configuration does not use the restored training protocol')
            if study.get('profile') == 'pilot' and (
                config.get('max_epochs') != 13000 or config.get('num_actors') != 12000
                or config.get('horizon_length') != 32
            ):
                raise ValueError('DexMachina full-run cap or rollout configuration is incorrect')
            for key, value in (
                ('checkpoint_selection', endpoint['selection']), ('checkpoint_epoch', endpoint['epoch']),
                ('completed_training_updates', endpoint['completed_updates']),
                ('training_epoch_cap', config['max_epochs']),
            ):
                if protocol.get(key) != value:
                    raise ValueError(f'DexMachina evaluation provenance differs for {key}')
    if not metadata.get("run_id"):
        raise ValueError(f"No W&B run id in {env_path}")

    import wandb

    init_kwargs = {
        "project": metadata.get("project") or args.wandb_project,
        "id": metadata["run_id"],
        "resume": "allow",
        "name": metadata.get("run_name"),
        "reinit": True,
    }
    if metadata.get("entity"):
        init_kwargs["entity"] = metadata["entity"]
    run = wandb.init(**init_kwargs)
    try:
        run.log(metrics)
        for key, value in metrics.items():
            run.summary[key] = value
        run.summary[f"{prefix}/stats_path"] = str(stats_path)
        if checkpoint is not None:
            run.summary[f'{prefix}/checkpoint'] = str(checkpoint)
        if protocol is not None:
            run.summary[f'{prefix}/protocol'] = protocol['protocol']
            run.summary[f'{prefix}/protocol_path'] = str(protocol_path)
            run.summary[f'{prefix}/checkpoint_sha256'] = protocol['checkpoint_sha256']
        if study_protocol in ('v2', 'dexmachina'):
            run.summary['study/status'] = 'completed'
            run.summary['study/protocol'] = study_protocol
            run.summary['study/profile'] = study['profile']
            run.summary['study/include_in_final_results'] = study['profile'] == 'pilot'
            run.summary['study/training_seed_count'] = 1
            run.summary[f'{prefix}/evaluation_seed'] = protocol['evaluation_seed']
            run.summary[f'{prefix}/initial_state_distribution'] = reset['initial_state_distribution']
            run.summary[f'{prefix}/reset_state_sha256'] = reset['state_file_sha256']
            run.summary[f'{prefix}/object_state_sampling'] = protocol['object_state_sampling']
            if study_protocol == 'dexmachina':
                run.summary[f'{prefix}/checkpoint_selection'] = endpoint['selection']
                run.summary[f'{prefix}/checkpoint_epoch'] = endpoint['epoch']
                run.summary['study/completed_updates'] = endpoint['completed_updates']
                run.summary['study/environment_transitions'] = endpoint['environment_transitions']
                run.summary['study/max_epochs_cap'] = config['max_epochs']
                run.summary['study/stop_reason'] = endpoint.get('stop_reason', 'not_recorded')
    finally:
        run.finish()


if __name__ == "__main__":
    main()
