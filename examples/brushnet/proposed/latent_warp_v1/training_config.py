"""YAML configuration, CLI precedence and launcher bridge (no GPU work)."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import sys

import yaml

DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'run1.yaml'
GROUPS = {'paths', 'data', 'training', 'diffusion', 'shared_noise', 'latent_warp', 'execution', 'logging', 'runtime'}
ENV_ARGS = {
    'PRETRAINED_MODEL': 'pretrained_model_name_or_path', 'IMAGE_ENCODER': 'image_encoder_name_or_path',
    'DATASET_ROOT': 'coco_train_root', 'TRAIN_SPLIT': 'train_split',
    'TEACHER_FLOW_ROOT': 'teacher_flow_root', 'OUTPUT_DIR': 'output_dir',
    'MIXED_PRECISION': 'mixed_precision', 'RESOLUTION': 'resolution',
    'CLIP_LENGTH': 'clip_length', 'CLIP_STRIDE': 'clip_stride',
    'CLIPS_PER_DEVICE': 'train_batch_size', 'GRADIENT_ACCUMULATION_STEPS': 'gradient_accumulation_steps',
    'MAX_TRAIN_STEPS': 'max_train_steps', 'CHECKPOINTING_STEPS': 'checkpointing_steps',
    'LEARNING_RATE': 'learning_rate', 'LR_SCHEDULER': 'lr_scheduler', 'LR_WARMUP_STEPS': 'lr_warmup_steps',
    'SHARED_BG_NOISE_STRENGTH': 'shared_bg_noise_strength',
    'SEQUENCE_SHARED_NOISE_REFRESH': 'sequence_shared_noise_refresh',
    'LATENT_WARP_DROP_PROB': 'latent_warp_drop_prob', 'LATENT_WARP_X0_CLAMP': 'latent_warp_x0_clamp',
    'SEED': 'seed', 'DATALOADER_NUM_WORKERS': 'dataloader_num_workers',
}


def read_config(path, seen=()):
    path = Path(path).expanduser().resolve()
    if path in seen:
        raise ValueError('Config inheritance cycle: ' + str(path))
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('Config must be a YAML mapping')
    unknown = set(data) - GROUPS - {'extends'}
    if unknown:
        raise ValueError('Unknown config sections: ' + ', '.join(sorted(unknown)))
    result = read_config(path.parent / data['extends'], seen + (path,)) if 'extends' in data else {}
    for section, values in data.items():
        if section == 'extends':
            continue
        if not isinstance(values, dict):
            raise ValueError(section + ' must be a mapping')
        result.setdefault(section, {}).update(values)
    return result


def parse_configured_args(parser, input_args=None):
    parser.add_argument('--config', default=os.environ.get('CONFIG', str(DEFAULT_CONFIG)))
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', default=os.environ.get('CONFIG', str(DEFAULT_CONFIG)))
    config_path = probe.parse_known_args(input_args)[0].config
    try:
        config = read_config(config_path)
        defaults = {}
        for group, values in config.items():
            if group == 'runtime':
                continue
            for key, value in values.items():
                if key in defaults:
                    raise ValueError('Duplicate setting: ' + key)
                defaults[key] = value
        for env, dest in ENV_ARGS.items():
            if env in os.environ:
                defaults[dest] = os.environ[env]
        if os.environ.get('INITIAL_CHECKPOINT'):
            root = Path(os.environ['INITIAL_CHECKPOINT'])
            defaults.update(brushnet_model_name_or_path=str(root/'brushnet'),
                            pretrained_ip_adapter_path=str(root/'ipadapter/model.safetensors'),
                            fusion_module_path=str(root/'ipadapter/fusion_module.safetensors'))
        actions = {a.dest: a for a in parser._actions}
        for key, value in defaults.items():
            if key not in actions or key in {'config', 'help', 'hub_token'}:
                raise ValueError('Unknown or unsupported config setting: ' + key)
            action = actions[key]
            if value is None:
                if action.default is not None:
                    raise ValueError(key + ' cannot be null')
                continue
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(value, bool):
                    raise ValueError(key + ' must be a YAML boolean')
            elif action.nargs in ('*', '+'):
                if not isinstance(value, list):
                    raise ValueError(key + ' must be a list')
                value = [action.type(v) if action.type else v for v in value]
            elif action.type:
                if isinstance(value, (dict, list, bool)):
                    raise ValueError('Invalid type for ' + key)
                if action.type is int and isinstance(value, float) and not value.is_integer():
                    raise ValueError(key + ' must be an integer')
                value = action.type(value)
            if action.choices and value not in action.choices:
                raise ValueError('Invalid choice for ' + key + ': ' + str(value))
            defaults[key] = value
        runtime = config.get('runtime', {})
        if set(runtime) - {'accelerate_bin', 'cuda_visible_devices', 'allow_shared_gpu'}:
            raise ValueError('Unknown runtime setting')
        if not isinstance(runtime.get('allow_shared_gpu', False), bool):
            raise ValueError('runtime.allow_shared_gpu must be boolean')
        parser.set_defaults(**defaults)
    except (ValueError, TypeError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    args = parser.parse_args(input_args)
    args.config = str(Path(config_path).expanduser().resolve())
    args._runtime_config = runtime
    return args


def verify_diffusion(args, scheduler):
    for key in ('num_train_timesteps', 'beta_schedule', 'beta_start', 'beta_end', 'prediction_type'):
        expected = getattr(args, 'expected_' + key)
        actual = getattr(scheduler.config, key)
        if actual != expected:
            raise ValueError('Diffusion config mismatch: {} expected {!r}, loaded {!r}'.format(key, expected, actual))


def save_resolved_config(args, scheduler, accelerator):
    # JSON conversion normalizes scheduler tuples/FrozenDict before safe YAML output.
    settings = {k: v for k, v in vars(args).items() if not k.startswith('_') and k != 'hub_token'}
    payload = {'arguments': settings, 'diffusion_actual': dict(scheduler.config),
               'runtime_actual': {'num_processes': accelerator.num_processes,
                                  'mixed_precision': accelerator.mixed_precision,
                                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                  'effective_batch_clips': accelerator.num_processes * args.train_batch_size * args.gradient_accumulation_steps,
                                  'effective_batch_frames': accelerator.num_processes * args.train_batch_size * args.gradient_accumulation_steps * args.clip_length}}
    path = Path(args.output_dir) / 'resolved_config.yaml'
    path.write_text(yaml.safe_dump(json.loads(json.dumps(payload)), sort_keys=False))


def launcher_settings(argv):
    # Trainer imports may print diagnostics; keep stdout strictly shell assignments.
    with contextlib.redirect_stdout(sys.stderr):
        from train_brushnet_latent_warp_v1 import parse_args
        args = parse_args(argv, validate_paths=False)
    runtime = args._runtime_config
    values = {env: getattr(args, dest) for env, dest in ENV_ARGS.items()}
    values.update(CONFIG=args.config,
                  BRUSHNET_CHECKPOINT=args.brushnet_model_name_or_path,
                  IPADAPTER_CHECKPOINT=args.pretrained_ip_adapter_path,
                  FUSION_CHECKPOINT=args.fusion_module_path,
                  ACCELERATE_BIN=os.environ.get('ACCELERATE_BIN', runtime.get('accelerate_bin', 'accelerate')),
                  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', runtime.get('cuda_visible_devices') or ''),
                  ALLOW_SHARED_GPU=os.environ.get('ALLOW_SHARED_GPU', str(int(runtime.get('allow_shared_gpu', False)))))
    for key, value in values.items():
        print('{}={}'.format(key, shlex.quote(str(value))))


if __name__ == '__main__':
    launcher_settings(sys.argv[1:])
