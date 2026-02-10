import argparse
from trainer import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['train', 'test', 'save'], required=True, help='Task to perform: train, test, or save the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, help='Path to save the model (required for save task)')
    args = parser.parse_args()

    trainer = TrainerModule(args.config)

    if args.task == 'test':
        trainer.test()
    elif args.task == 'save':
        if not args.model_path:
            raise ValueError("Model path must be provided for saving the model.")
        trainer.save_model(args.model_path)
    else:  # Default to training if task is 'train'
        if args.task != 'train':
            raise ValueError("Invalid task. Use 'train', 'test', or 'save'.")
        trainer.train()