import subprocess

if __name__ == "__main__":

    for seed in [1, 2, 3]:
        for byz_iter in [0.4]:
            for byz in [1]:
                for attack in ['sf', 'lf']:
                    for agg in ['weighted_rfa', 'weighted_cwmed']:
                        for optimizer in ['mu2sgd', 'momentum', 'sgd']:
                                if optimizer == "mu2sgd":
                                    gradient_momentum = 0.25
                                else:
                                    gradient_momentum = 0.9
                                command = [
                                    "python", "main.py",
                                    "--sampler", "id",
                                    "--dataset", "mnist",
                                    "--model", "conv_mnist",
                                    "--epoch_num", "1",
                                    "--eval_interval", "150",
                                    "--optimizer", optimizer,
                                    "--learning_rate", "0.01",
                                    "--gradient_momentum", f"{gradient_momentum}",
                                    "--query_point_momentum", f"0.1",
                                    "--batch_size", "16",
                                    "--seed", str(seed),
                                    "--agg", agg,
                                    "--workers_num", "9",
                                    "--byzantine_num", str(byz),
                                    "--use_wandb",
                                    "--weight_decay", "0.0",
                                    "--lambda_byz", str(byz_iter),
                                    '--experiment_name', "Asynchronous Byzantine Experiment - MNIST"
                                ]

                                if attack is not None:
                                    command.extend(["--attack", attack])

                                subprocess.run(command)




