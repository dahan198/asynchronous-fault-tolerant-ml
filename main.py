import argparse
from torch.utils.data import DataLoader
from weight4robustness.dataset import DATASET_REGISTRY
from weight4robustness.model import MODEL_REGISTRY
from weight4robustness.trainer.async_trainer import AsyncTrainer
from weight4robustness.optimizer import OPTIMIZER_REGISTRY
from weight4robustness.worker_sampler import SAMPLER
from weight4robustness.utils import set_seed, get_device
from weight4robustness.aggregators import get_aggregator
from weight4robustness.worker import WorkerMomentum, WorkerSTORM, WorkerSGD
from weight4robustness.attacks import get_attack
import math


def main():
    """Main entry point for the training script."""

    parser = argparse.ArgumentParser(description="Training script for fault-tolerant machine learning models.")

    # Argument definitions
    parser.add_argument('--agg', type=str, default='weighted_rfa',
                        choices=['cwmed', 'rfa', 'weighted_avg', 'weighted_rfa', 'weighted_cwmed'],
                        help='Aggregation rule for robust distributed training.')
    parser.add_argument('--attack', type=str, default='lf', choices=['lf', 'sf', 'ipm', 'alie'],
                        help='Type of attack for Byzantine robustness.')
    parser.add_argument('--sampler', type=str, default='id2', choices=['id', 'uniform', 'id2'],
                        help='Sampler strategy.')
    parser.add_argument('--workers_num', type=int, default=17, help='Number of workers.')
    parser.add_argument('--byzantine_num', type=int, default=8, help='Number of Byzantine workers.')
    parser.add_argument('--lambda_byz', type=float, default=0.4, help='Fraction of Byzantine iterations.')
    parser.add_argument('--config_folder_path', type=str, default='./config', help='Path to configuration folder.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=DATASET_REGISTRY.keys(),
                        help='Dataset to be used for training.')
    parser.add_argument('--model', type=str, default='conv_mnist', choices=MODEL_REGISTRY.keys(),
                        help='Model architecture to be used.')
    parser.add_argument('--epoch_num', type=int, default=1, help='Number of epochs for training.')
    parser.add_argument('--eval_interval', type=int, default=160, help='Interval for evaluation during training.')
    parser.add_argument('--optimizer', type=str, default='mu2sgd',
                        choices=['mu2sgd', 'momentum', 'sgd'], help='Optimizer to be used.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--gradient_momentum', type=float, default=0.25,
                        help='Momentum for gradient updates (if applicable).')
    parser.add_argument('--query_point_momentum', type=float, default=0.1,
                        help='Momentum for query points.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging.')
    parser.add_argument('--ipm_epsilon', type=float, default=0.1, help='Epsilon parameter for IPM attack.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for regularization.')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment.')

    # Parse arguments
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    # Initialize attack
    attack = get_attack(args.attack, device=device, epsilon=args.ipm_epsilon,
                        workers_num=args.workers_num, byzantine_num=args.byzantine_num) if args.attack else None

    # Initialize aggregator
    aggregator = get_aggregator(args.agg, args.workers_num, args.byzantine_num)

    # Load dataset and model
    dataset = DATASET_REGISTRY[args.dataset]()
    model = MODEL_REGISTRY[args.model]().to(device)
    train_dataloader = DataLoader(dataset.trainset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset.testset, batch_size=args.batch_size, shuffle=False)

    # Configure optimizer and workers
    if args.optimizer == "momentum":
        optimizer = OPTIMIZER_REGISTRY["sgd"]
        workers = [WorkerMomentum(beta=args.gradient_momentum) for _ in range(args.workers_num)]
        optimizer_params = {"lr": args.learning_rate, "momentum": 0.0, "weight_decay": args.weight_decay}
    elif args.optimizer == "mu2sgd":
        optimizer = OPTIMIZER_REGISTRY["anytime_sgd"]
        workers = [WorkerSTORM(beta=args.gradient_momentum)
                   for _ in range(args.workers_num)]
        optimizer_params = {"lr": args.learning_rate, "gamma": args.query_point_momentum,
                            "weight_decay": args.weight_decay}
    else:
        optimizer = OPTIMIZER_REGISTRY["sgd"]
        workers = [WorkerSGD() for _ in range(args.workers_num)]
        optimizer_params = {"lr": args.learning_rate, "momentum": 0.0, "weight_decay": args.weight_decay}

    optimizer = optimizer(model.parameters(), **optimizer_params)

    # Initialize trainer
    byz_interval = math.ceil(1 / args.lambda_byz) if args.lambda_byz > 0 else 0.0
    trainer = AsyncTrainer(model, optimizer, train_dataloader, test_dataloader, args,
                           aggregator, workers, args.byzantine_num, attack, byz_interval,
                           args.experiment_name, device, SAMPLER[args.sampler])

    # Start training
    trainer.train(epoch_num=args.epoch_num, eval_interval=args.eval_interval)


if __name__ == "__main__":
    main()
