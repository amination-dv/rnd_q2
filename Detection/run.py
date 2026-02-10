"""
CLI entry-point for ILI component detection.

Usage:
    python run.py --task train --config config.yaml
    python run.py --task test  --config config.yaml
    python run.py --task save  --config config.yaml --model_path ./exported/model.pth
"""

import argparse
from trainer import TrainerModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ILI Component Detection — Train / Test / Save')
    parser.add_argument('--task', choices=['train', 'test', 'save'], required=True,
                        help='Task to perform')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Destination path (required for --task save)')
    args = parser.parse_args()

    module = TrainerModule(args.config)

    if args.task == 'train':
        module.train()
    elif args.task == 'test':
        module.test()
    elif args.task == 'save':
        if not args.model_path:
            raise ValueError("--model_path is required when --task is 'save'")
        module.save_model(args.model_path)
