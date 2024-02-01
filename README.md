# FedShift

## Installation

```
#Python 3.9.4

pip install -U pip
pip install -r requirements.txt

```

## Project Directory Layout

- `main.py`
  - The core script for the entire training workflow.
  - Implements the FedShift Process.
- `train.py`
  - Handles the training operations for the client.
- `quantization.py`
  - Includes algorithms for both quantization and dequantization.
- `models.py`
  - Contains the foundational models used in the project.
- `data_loader.py`
  - Loader for datasets (CIFAR10, CIFAR100) with support for non-iid data distribution.
- `utils.py`
  - A collection of utility functions.

## How To Run

- Vanilla Mode with no quantization

```
#FedAvg (by default)
python main.py
#FedProx
python main.py --use_fedprox
#SCAFFOLD
python main.py --use_scaffold
```

- With quantization

```
#FedAvg (by default)
python main.py --use_quantization
#FedProx
python main.py --use_fedprox --use_quantization
#SCAFFOLD
python main.py --use_scaffold --use_quantization
```

- With quantization + FedShift

```
#FedAvg
python main.py --use_quantization --use_fedshift
#FedProx
python main.py --use_fedprox --use_quantization --use_fedshift
#SCAFFOLD
python main.py --use_scaffold --use_quantization --use_fedshift
```

- Changing quantization method

```
#Non-Uniform quantization (by default)
python main.py --use_quantization --use_fedshift --no_uniform_q

#Uniform quantization
python main.py --use_quantization --use_fedshift --use_uniform_q

```

- Changing dataset

```
#Uniform quantization
python main.py --use_quantization --use_fedshift --data_type CIFAR10

#Uniform quantization
python main.py --use_quantization --use_fedshift --data_type CIFAR100
```

- Changing non-IID

```
#Inferior+Superior non-iid (by default)
python main.py --use_quantization --use_fedshift

#Dirichlet non-iid
python main.py --use_quantization --use_fedshift --dir_alpha 0.1
```

- Changing seed

```
python main.py --seed 42
```
