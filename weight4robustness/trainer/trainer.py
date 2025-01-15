import torch
import torch.nn as nn
import torchmetrics
import os
import json
import csv
import wandb


class Trainer:
    """
    A class to manage the training process of a machine learning model, including training, evaluation,
    logging, checkpointing, and metrics tracking.
    """

    def __init__(self, model, optimizer, train_dataloader, test_dataloader, params, aggregator, workers,
                 byzantine_num, attack, device, experiment_name=None):
        """
        Initializes the Trainer class with the given model, optimizer, data loaders, and parameters.

        Args:
            model (nn.Module): The neural network model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
            params (argparse.Namespace): Parsed command-line parameters.
            aggregator (callable): Aggregation function for distributed updates.
            workers (list): List of worker instances.
            byzantine_num (int): Number of Byzantine workers.
            attack (callable): Attack strategy for Byzantine robustness.
            device (torch.device): Device to run training on (CPU or GPU).
            experiment_name (str, optional): Name of the experiment for logging purposes. Defaults to None.
        """
        self.workers_num = len(workers)
        self.device = device
        self.byzantine_num = byzantine_num
        self.honest_num = self.workers_num - self.byzantine_num
        self.attack = attack
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.aggregator = aggregator
        self.workers = workers
        self.experiment_name = experiment_name
        self.params = params.__dict__
        self.param_shapes = [p.shape for p in model.parameters()]
        self.split_sizes = [torch.prod(torch.tensor(shape)).item() for shape in self.param_shapes]
        self.checkpoint_path = None
        self.log_file_path = None
        self.run_directory = "./"
        self.use_wandb = self.params["use_wandb"]
        self.wandb = wandb
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)
        self.best_accuracy = 0.0
        self.metrics_data = []
        self.iter_results = {
            "iteration": 0,
            "train_loss": 0,
            "train_acc": 0,
            "test_loss": 0,
            "test_acc": 0,
            **params.__dict__
        }

    def train(self):
        """
        Prepares the training environment, including creating directories for logging and checkpoints,
        and saving experiment parameters.
        """
        # Create results directory
        results_dir = 'results' if self.experiment_name is None else f'results-{self.experiment_name}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Determine the next run folder
        existing_folders = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        next_folder_num = len(existing_folders) + 1
        run_directory = os.path.join(results_dir, f'run{next_folder_num}')
        os.makedirs(run_directory)
        self.checkpoint_path = os.path.join(run_directory, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.log_file_path = os.path.join(run_directory, 'metrics_log.txt')
        self.run_directory = run_directory

        # Save parameters to a JSON file
        params_file_path = os.path.join(run_directory, 'params.json')
        with open(params_file_path, 'w') as params_file:
            json.dump(self.params, params_file, indent=4)

    def evaluate(self):
        """
        Evaluates the model on the test dataset.

        Returns:
            tuple: A tuple containing the test accuracy and average test loss.
        """
        self.model.eval()
        self.accuracy_metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                self.accuracy_metric.update(predictions, labels)

        final_accuracy = self.accuracy_metric.compute()
        test_loss = total_loss / len(self.test_dataloader)
        self.model.train()
        return final_accuracy, test_loss

    def make_evaluation_step(self, iteration, total_iterations, eval_interval, running_loss, metrics_table, iter_title):
        """
        Performs an evaluation step during training, logs metrics, and saves the model if it achieves
        the best accuracy.

        Args:
            iteration (int): Current iteration number.
            total_iterations (int): Total number of iterations.
            eval_interval (int): Interval for evaluation.
            running_loss (float): Cumulative loss over the interval.
            metrics_table (prettytable.PrettyTable): Table for logging metrics.
            iter_title (str): Title for the iteration in metrics logging.
        """
        train_accuracy = self.accuracy_metric.compute()
        average_loss = running_loss / eval_interval
        test_accuracy, test_loss = self.evaluate()

        # Log metrics to the table and file
        metrics_table.add_row(
            [f"{iteration + 1}/{total_iterations}", f"{average_loss:.4f}", f"{train_accuracy:.2%}",
             f"{test_loss:.4f}", f"{test_accuracy:.2%}"]
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"{metrics_table.get_string()}\n\n")

        print(metrics_table)
        metrics_table.clear_rows()

        # Append metrics for later saving
        self.metrics_data.append({
            iter_title: iteration,
            "Train Loss": average_loss,
            "Train Accuracy": train_accuracy.item(),
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy.item()
        })

        # Log metrics to wandb
        if self.use_wandb:
            self.wandb.log({
                "Train Loss": average_loss,
                "Train Accuracy": train_accuracy,
                "Test Loss": test_loss,
                "Test Accuracy": test_accuracy
            })

        # Save the model if the test accuracy is the best so far
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, "best.pth"))

        self.accuracy_metric.reset()

    def save_metrics_and_params(self):
        """
        Saves the collected metrics and parameters to a CSV file.
        """
        csv_file_path = os.path.join(self.run_directory, 'results.csv')
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        if not self.metrics_data:
            print("No metrics data to save.")
            return

        combined_rows = [{**metrics, **self.params} for metrics in self.metrics_data]
        fieldnames = list(combined_rows[0].keys())

        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    def make_optimization_step(self, inputs, targets, first_step=False):
        """
        Placeholder method for performing an optimization step.
        """
        pass

    def get_worker_gradient(self, inputs, labels):
        """
        Computes the gradient for a given input and label batch.

        Args:
            inputs (torch.Tensor): Input data.
            labels (torch.Tensor): Corresponding labels.

        Returns:
            tuple: A tuple containing the computed gradient, loss value, and model output.
        """
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        loss.backward()
        gradient = torch.cat([param.grad.detach().clone().flatten() for param in self.model.parameters()])
        return gradient, loss.item(), output
