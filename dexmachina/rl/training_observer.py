"""Training records of runs with --ablation_protocol v2|dexmachina, without changing the inherited PPO update or schedule:
the initial network (params/initialization.json, params/initial_model.pth) and the update and transition counts in W&B."""

from rl_games.common.algo_observer import IsaacAlgoObserver


class TrainingObserver(IsaacAlgoObserver):
    def after_init(self, algo):
        super().after_init(algo)
        protocol = algo.config.get('ablation_protocol')
        if protocol not in ('v2', 'dexmachina'):
            return
        import json
        from pathlib import Path
        import torch
        import wandb
        from dexmachina.rl.ablation_initialization import parameter_hash, seed_all

        network = algo.model.a2c_network
        metadata = {
            'protocol': protocol,
            'initialization_seed': algo.config.get('initialization_seed', algo.config.get('training_seed')),
            'actor_sha256': parameter_hash(network, 'actor'),
            'critic_trunk_sha256': parameter_hash(network, 'critic_trunk'),
            'value_sha256': parameter_hash(network, 'value'),
            'rollout_seed': algo.config.get('rollout_seed'),
            'rng_protocol': ('independent_named_layer_initialization; reset_shared_rollout_stream_after_model_build'
                             if protocol == 'v2' else 'native_rl_games_initialization; no_post_model_reseed'),
        }
        run_dir = Path(algo.config['train_dir']) / algo.config['full_experiment_name']
        (run_dir / 'params').mkdir(parents=True, exist_ok=True)
        (run_dir / 'params' / 'initialization.json').write_text(json.dumps(metadata, indent=2) + '\n')
        torch.save({k: v.detach().cpu() for k, v in network.state_dict().items()},
                   run_dir / 'params' / 'initial_model.pth')
        if wandb.run is not None:
            wandb.run.summary.update({f'initialization/{key}': value for key, value in metadata.items()})
        if protocol == 'v2':
            seed_all(algo.config['rollout_seed'])

    def wandb_after_print_stats(self, frame, epoch_num, total_time):
        metrics = super().wandb_after_print_stats(frame, epoch_num, total_time)
        if self.algo.config.get('ablation_protocol') == 'dexmachina':
            env = self.algo.vec_env.env.unwrapped
            metrics.update({'study/completed_updates': epoch_num,
                            'study/environment_transitions': env.environment_transitions})
        return metrics
