

# Weight for Robustness: A Comprehensive Approach towards Optimal Fault-Tolerant Asynchronous ML

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue.svg)](#) [![arXiv](https://img.shields.io/badge/arXiv-2304.04169-B31B1B.svg)](https://arxiv.org/abs/2501.09621) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains the official implementation of the paper **"Weight for Robustness: A Comprehensive Approach towards Optimal Fault-Tolerant Asynchronous ML"**, authored by Tehila Dahan and Kfir Y. Levy, and accepted at **NeurIPS 2024**. 

The repository provides a modular framework for training machine learning models in asynchronous distributed settings while ensuring robustness against Byzantine faults using advanced aggregation techniques and attack-resistant strategies.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dahan198/asynchronous-fault-tolerant-ml.git
cd asynchronous-fault-tolerant-ml
```

### 2. Install Dependencies

First, ensure that PyTorch is installed. You can install it by selecting the appropriate command based on your environment from [PyTorch's official website](https://pytorch.org/get-started/locally/).

#### Install Other Dependencies

After installing PyTorch, install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

---

## **Usage**

### **Running an Experiment**

To train an experiment, run the following command:

```bash
python run.py
```

---

## Results and Logging

- **Weights & Biases (wandb)**: This repository supports logging with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization. Ensure you have a `wandb.yaml` file in the `config` directory with your project and entity name.
  
  Example `wandb.yaml`:
  ```yaml
  project: "Weight4Robustness"
  entity: "your-wandb-username"
  ```

---

## Citation

If you find this code useful in your research, please consider citing our paper:

```
@inproceedings{weight4robustness2024,
  title={Weight for Robustness: A Comprehensive Approach towards Optimal Fault-Tolerant Asynchronous ML},
  author={Tehila Dahan and Kfir Yehuda Levy},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.





