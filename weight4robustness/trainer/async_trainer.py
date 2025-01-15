import torch
import os
from .trainer import Trainer
from prettytable import PrettyTable
from tqdm import tqdm
import yaml


class AsyncTrainer(Trainer):
    """
    Implements an asynchronous trainer for distributed training with Byzantine robustness.
    This trainer handles asynchronous updates, sampling of workers, and applying attacks in
    a distributed setting.
    """

    def __init__(self, model, optimizer, train_dataloader, test_dataloader, params, aggregator, workers,
                 byzantine_num, attack, byz_iterations, experiment_name, device, sampler):
        """
        Initializes the AsyncTrainer with the given model, optimizer, data loaders, and additional parameters.

        Args:
            model (nn.Module): The neural network model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer for model parameter updates.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
            params (argparse.Namespace): Parsed command-line parameters.
            aggregator (callable): Aggregation function for robust distributed updates.
            workers (list): List of worker instances.
            byzantine_num (int): Number of Byzantine workers.
            attack (callable): Attack strategy to be applied by Byzantine workers.
            byz_iterations (int): Number of iterations after which Byzantine workers are sampled.
            experiment_name (str, optional): Name of the experiment. Defaults to None.
            device (torch.device): Device to run training on (CPU or GPU).
            sampler (callable): Function to sample workers during training.
        """
        super().__init__(model, optimizer, train_dataloader, test_dataloader, params, aggregator, workers,
                         byzantine_num, attack, device, experiment_name)
        self.train_iteration = 0
        self.honest_num = len(workers) - byzantine_num
        self.byz_iterations = byz_iterations
        self.byzantine_ids = list(range(self.honest_num, self.workers_num))
        self.weights = torch.zeros((self.workers_num, 1)).to(self.device)
        self.sampler = sampler
        total_size = sum(param.numel() for param in model.parameters())
        self.current_workers_momentums = torch.zeros((self.workers_num, total_size), device=device)
        self.next_workers_momentums = torch.zeros((self.workers_num, total_size), device=device)

        # Initialize Weights & Biases (wandb) if enabled
        if self.params["use_wandb"]:
            with open(os.path.join(params.config_folder_path, "wandb.yaml"), 'r') as file:
                self.wandb_conf = yaml.safe_load(file)
            algorithm = {'mu2sgd': 'Mu2SGD', 'momentum': 'Momentum', 'sgd': 'SGD'}
            project = self.wandb_conf["project"] if experiment_name is None else experiment_name
            self.wandb.init(
                project=project,
                entity=self.wandb_conf["entity"],
                name=f"{algorithm[self.params['optimizer']]}--{self.params['dataset']}--{self.params['model']}"
                     f"--{self.params['attack']}--{self.params['agg']}--LR: {self.params['learning_rate']}--"
                     f"Seed: {self.params['seed']}--Batch: {self.params['batch_size']}--"
                     f"Workers: {self.params['workers_num']}--Byz: {self.params['byzantine_num']}"
            )
            self.wandb.config.update(self.params)

    def train(self, epoch_num: int = 100, eval_interval: int = None):
        """
        Trains the model asynchronously for the specified number of epochs, with optional evaluation intervals.

        Args:
            epoch_num (int): Number of epochs to train. Defaults to 100.
            eval_interval (int, optional): Interval for performing evaluations during training. Defaults to None.
        """
        super().train()
        self.model.train()
        self.accuracy_metric.reset()
        self.make_first_optimization_step()

        metrics_table = PrettyTable()
        iter_title = "Epoch" if eval_interval is None else "Iteration"
        metrics_table.field_names = [iter_title, "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]

        total_iterations = epoch_num if eval_interval is None \
            else int(epoch_num * (len(self.train_dataloader) / eval_interval))
        running_loss = 0.0
        self.train_iteration = 0

        for epoch in range(epoch_num):

            for inputs, labels in tqdm(self.train_dataloader):
                self.train_iteration += 1
                outputs, loss = self.make_optimization_step(inputs, labels)

                if loss is not None:
                    running_loss += loss
                    predictions = torch.argmax(outputs, dim=1)
                    self.accuracy_metric.update(predictions, labels.to(self.device))

                if eval_interval is not None:
                    if self.train_iteration % eval_interval == 0:
                        self.make_evaluation_step(
                            (self.train_iteration // eval_interval) - 1, total_iterations, eval_interval,
                            running_loss, metrics_table, "Iteration"
                        )
                        self.train_iteration = 0
                        running_loss = 0.0
            if eval_interval is None:
                self.make_evaluation_step(epoch,
                                          epoch_num,
                                          len(self.train_dataloader),
                                          running_loss, metrics_table,
                                          "Epoch")
                self.train_iteration = 0
                running_loss = 0.0

        self.save_metrics_and_params()
        print('Finished Training')

    def make_first_optimization_step(self):
        """
        Performs the initial optimization step for all workers.
        This step initializes the momentum for each worker.
        """
        for i, (inputs, targets) in enumerate(self.train_dataloader):
            if i == self.workers_num:
                break
            self.weights[i] += 1
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if i < self.honest_num:  # Honest worker
                if self.workers[i].two_passes:  # STORM workers
                    gradient, _, _ = self.get_worker_gradient(inputs, targets)
                    self.workers[i].compute_estimator(gradient)
                else:  # Momentum workers
                    gradient, _, _ = self.get_worker_gradient(inputs, targets)
                    self.workers[i].step(gradient)
                self.current_workers_momentums[i] = self.workers[i].momentum.clone()
                self.next_workers_momentums[i] = self.workers[i].momentum.clone()
            else:  # Byzantine worker
                honest_momentums = self.current_workers_momentums[:self.honest_num].clone()
                byzantine_momentum = self.calc_byzantine_momentum(inputs, targets, i, honest_momentums)
                self.current_workers_momentums[i] = byzantine_momentum.clone()
                self.next_workers_momentums[i] = byzantine_momentum.clone()

    def make_optimization_step(self, inputs, targets, first_step=False, make_opt_step=True):
        """
        Performs an asynchronous optimization step by selecting a worker, computing gradients,
        and updating the model parameters.

        Args:
            inputs (torch.Tensor): Input data.
            targets (torch.Tensor): Target labels.
            first_step (bool, optional): Indicates whether this is the first optimization step. Defaults to False.
            make_opt_step (bool, optional): Whether to perform the optimizer step. Defaults to True.

        Returns:
            tuple: The model outputs and the loss value.
        """
        worker_id = self.get_next_worker()
        self.weights[worker_id] += 1
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.workers[worker_id].two_passes:  # STORM workers
            loss, outputs, _ = self.workers_optimization_step(inputs, targets, worker_id)
            self.optimizer.step()

            if worker_id >= self.honest_num and self.attack.__class__.__name__ == "LabelFlippingAttack":
                gradient, _, _ = self.get_worker_gradient(inputs, 9 - targets)
                self.workers[worker_id].compute_estimator(gradient)
            elif not (self.attack.__class__.__name__ in ["ALittleIsEnoughAttack", "IPMAttack"] and
                      worker_id >= self.honest_num):
                gradient, _, _ = self.get_worker_gradient(inputs, targets)
                self.workers[worker_id].compute_estimator(gradient)
        else:  # Momentum workers
            loss, outputs, _ = self.workers_optimization_step(inputs, targets, worker_id)
            self.optimizer.step()

        return outputs, loss

    def workers_optimization_step(self, inputs, labels, worker_id):
        """
        Performs the optimization step for the selected worker, aggregates the momentums,
        and updates the model parameters.

        Args:
            inputs (torch.Tensor): Input data.
            labels (torch.Tensor): Target labels.
            worker_id (int): ID of the selected worker.

        Returns:
            tuple: The loss value, model outputs, and the stacked workers' momentums.
        """
        loss, outputs = None, None

        self.current_workers_momentums[worker_id] = self.next_workers_momentums[worker_id].clone()
        workers_momentum = self.current_workers_momentums.clone()

        if worker_id < self.honest_num:  # Honest worker
            loss, outputs = self.calc_honest_momentum(inputs, labels, worker_id)
            self.next_workers_momentums[worker_id] = self.workers[worker_id].momentum.clone()
        else:
            honest_momentums = self.current_workers_momentums[:self.honest_num].clone()
            byzantine_momentum = self.calc_byzantine_momentum(inputs, labels, worker_id, honest_momentums)
            self.next_workers_momentums[worker_id] = byzantine_momentum.clone()

        aggregated_momentum = self.aggregator(workers_momentum, self.weights)
        split_momentum = torch.split(aggregated_momentum, self.split_sizes)

        for i, p in enumerate(self.model.parameters()):
            p.grad = split_momentum[i].view(self.param_shapes[i]).clone()

        return loss, outputs, workers_momentum

    def calc_byzantine_momentum(self, inputs, labels, worker_id, honest_momentums):
        """
        Updates the momentum for a Byzantine worker using the specified attack strategy.

        Args:
            inputs (torch.Tensor): Input data.
            labels (torch.Tensor): Target labels.
            worker_id (int): ID of the Byzantine worker.
            honest_momentums (torch.Tensor): Tensor of honest workers' momentums.

        Returns:
            torch.Tensor: The updated Byzantine momentum.
        """
        byzantine_momentum = self.attack.apply(inputs=inputs,
                                               targets=labels,
                                               honest_updates=honest_momentums,
                                               worker=self.workers[worker_id],
                                               gradient_function=self.get_worker_gradient,
                                               weights=self.weights)
        return byzantine_momentum

    def calc_honest_momentum(self, inputs, labels, worker_id):
        """
        Updates the momentum for an honest worker by computing its gradient.

        Args:
            inputs (torch.Tensor): Input data.
            labels (torch.Tensor): Target labels.
            worker_id (int): ID of the honest worker.

        Returns:
            tuple: The loss value and model outputs.
        """
        gradient, loss, outputs = self.get_worker_gradient(inputs, labels)
        self.workers[worker_id].step(gradient)
        return loss, outputs

    def get_next_worker(self):
        """
        Samples the next worker based on the specified sampling strategy.

        Returns:
            int: The ID of the selected worker.
        """
        if self.train_iteration % self.byz_iterations == 0 and self.byz_iterations != 0 and self.byzantine_num > 0:
            return self.byzantine_ids[self.sampler(self.byzantine_num)]
        return self.sampler(self.honest_num)
