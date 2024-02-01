import argparse  # <-- Add this import
import copy
import multiprocessing
import os
import random
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import statistics
import numpy as np
import torch
import torch.nn as nn

from torchvision import models
from data_loader import load_cifar10_data, load_cifar100_data, load_FMNIST_data
from models import CNN2, CNNBN, FNN, ResNet18, CNN3
from quantization import dequantization
from train import client_training_process, evaluate
from utils import (
    client_selection,
    compute_KL_divergence,
    cosine_similarity,
    covariance,
    init_weights,
    l2_distance,
    set_seed,
)


import time


def main(args):
    set_seed(args.seed)
    # Initialize wandb
    # run_name = f"Landscape B{args.bits}_FP{int(args.use_fedprox)}_FS{int(args.use_fedshift)}_UQ{int(args.use_uniform_q)}_SF{int(args.use_scaffold)}"
    # run_name = f"IID Shifting Prob{args.shifting_prob}"
    # run_name = f"Landscape 32bit non-iid 10/100 clients"
    run_name = f" Dir_alpha {args.dir_alpha} B{args.bits}_FP{int(args.use_fedprox)}_FS{int(args.use_fedshift)}_UQ{int(args.use_uniform_q)}_SF{int(args.use_scaffold)}"
    # run_name = f"32+4 non-iid "
    print(args)
    # Hyperparameters
    num_clients = args.num_clients
    num_selected_clients = args.selected_clients

    epochs = 10
    batch_size = 50
    rounds = 400  # Number of communication rounds
    bits = args.bits  # <-- Use args.bits
    seed = args.seed
    use_fedprox = args.use_fedprox  # <-- Use args.use_fedprox
    use_fedshift = args.use_fedshift  # <-- Use args.use_fedshift
    use_uniform_q = args.use_uniform_q
    use_scaffold = args.use_scaffold
    use_quantization = args.use_quantization
    dir_alpha = args.dir_alpha
    dirichlet = True if dir_alpha else False
    mu = 0.01  # This is the coefficient for the proximal term, tune as needed
    mp = True
    iid = False
    num_of_inferior = 5
    data_type = args.data_type if args.data_type else "CIFAR10"

    # trainloaders, testloader = load_cifar10_data(batch_size, num_clients, iid=True)

    if data_type == "CIFAR10":
        data_ldr = load_cifar10_data
    else:
        data_ldr = load_cifar100_data

    trainloaders, testloader = data_ldr(
        batch_size,
        num_clients,
        iid=iid,
        shard_size=500,
        dirichlet=True,
        alpha=dir_alpha,
        seed=seed,
    )

    # Device initialization & model setup
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    # device = torch.device("cpu")
    if device.type == "cuda":
        multiprocessing.set_start_method("spawn", force=True)

    model_type = "CNN"
    if model_type == "CNN":
        if data_type == "CIFAR10":
            classes = 10
        else:
            classes = 100
        global_model = CNN2("CNN", 3, 32, 512, classes).to(device)
        alternative_model = CNN2("CNN", 3, 32, 512, classes).to(device)
    elif model_type == "FNN":
        global_model = FNN(
            name="FNN", input_dim=28 * 28 * 1, num_hiddens=128, num_classes=10
        ).to(device)
        alternative_model = FNN(
            name="FNN", input_dim=28 * 28 * 1, num_hiddens=128, num_classes=10
        ).to(device)

    init_weights(global_model, "kaiming", 1.0)

    global_control_variate = {
        k: torch.zeros_like(param).to("cpu")
        for k, param in global_model.named_parameters()
    }

    control_variates = [
        copy.deepcopy(global_control_variate) for _ in range(num_clients)
    ]
    for communication_round in range(rounds):
        # Randomly select a subset of clients
        selected_clients = client_selection(
            num_selected_clients, num_clients, mode="half-odd-even"
        )
        print(f"\n[Communication Round {communication_round + 1}/{rounds}]")
        print(f"Selected Clients: {selected_clients}")

        # Train each selected client model in parallel using multiprocessing pool
        if mp:
            arguments = [
                (
                    client_id,
                    copy.deepcopy(global_model).to("cpu"),
                    device,
                    trainloaders[client_id],
                    epochs,
                    bits,
                    use_fedprox,  # Add the use_fedprox flag to the arguments
                    use_uniform_q,
                    use_quantization,
                    mu,  # Add the mu parameter for the proximal term
                    seed,
                    use_scaffold,
                    copy.deepcopy(global_control_variate),
                    control_variates[client_id],
                )
                for idx, client_id in enumerate(selected_clients)
            ]

            with Pool(processes=5) as pool:
                results = pool.map(client_training_process, arguments)

            # Extract the model state dicts from the results

            client_updates = {
                client_id: (state_dict, codebook, local_control_variate)
                for client_id, (state_dict, codebook), local_control_variate in results
            }

        else:
            client_updates = {}

            for idx, client_id in enumerate(selected_clients):
                # Unpack arguments for the training process
                args_for_client = (
                    client_id,
                    copy.deepcopy(global_model).to("cpu"),
                    device,
                    trainloaders[client_id],
                    epochs,
                    bits,
                    use_fedprox,  # Add the use_fedprox flag to the arguments
                    use_uniform_q,
                    use_quantization,
                    mu,  # Add the mu parameter for the proximal term
                    seed,
                    use_scaffold,
                    copy.deepcopy(global_control_variate),
                    control_variates[client_id],
                )
                # Directly call the training process for each client
                (
                    client_id,
                    (state_dict, codebook),
                    local_control_variate,
                ) = client_training_process(args_for_client)

                client_updates[client_id] = (
                    state_dict,
                    codebook,
                    local_control_variate,
                )

        # Initialize dictionaries to hold intermediate values
        global_dict = global_model.state_dict()
        stacked_weights = {k: [] for k in global_dict.keys()}
        stacked_c_delta_weights = {k: [] for k in global_dict.keys()}
        c_delta_avg = {}
        superior_stacked_weights = {k: [] for k in global_dict.keys()}
        inferior_stacked_weights = {k: [] for k in global_dict.keys()}

        deq_weights_cache = {}

        for client_id in selected_clients:
            for k in global_dict.keys():
                if client_id % 2 == 1 and use_quantization:
                    if client_id not in deq_weights_cache:
                        deq_weights_cache[client_id] = {}

                    weight_dict = client_updates[client_id][0][k].long()
                    codebook = client_updates[client_id][1][k]

                    deq_weight = dequantization(
                        weight_dict, bits, codebook, use_uniform_q
                    )
                    deq_weights_cache[client_id][k] = deq_weight
                    stacked_weights[k].append(deq_weight.float().to("cpu"))
                    inferior_stacked_weights[k].append(deq_weight.float().to("cpu"))
                else:
                    stacked_weights[k].append(
                        client_updates[client_id][0][k].float().to("cpu")
                    )
                    superior_stacked_weights[k].append(
                        client_updates[client_id][0][k].float().to("cpu")
                    )
                if use_scaffold:
                    stacked_c_delta_weights[k].append(
                        client_updates[client_id][2][k].float().to("cpu")
                        - control_variates[client_id][k].float().to("cpu")
                    )
                    # update c_i with c+_i
                    control_variates[client_id][k] = (
                        client_updates[client_id][2][k].float().to("cpu")
                    )

        # Aggregate and compute mean of means
        for k, weights in stacked_weights.items():
            averaged_weights = torch.stack(weights, 0).mean(0)
            # Subtracting the mean to ensure the average is 0
            global_dict[k] = averaged_weights  # - averaged_weights.mean()

            if use_scaffold:
                c_delta_avg[k] = torch.stack(stacked_c_delta_weights[k], 0).mean(0)

                global_control_variate[k] += (
                    1.0 * len(selected_clients) / num_clients
                ) * c_delta_avg[k]

        alternative_model.load_state_dict(global_dict)
        alternative_model.to(device)

        # Evaluate the global model after each communication round
        print("Evaluating global model:")
        evaluate(alternative_model, device, testloader, communication_round, False)
        if use_fedshift:
            # Step 2: Calculate the mean of means (scalar) for each layer
            means_dict = {k: v.mean().item() for k, v in global_dict.items()}
            print("Before-shifting", means_dict)
            for k, v in global_dict.items():
                I = num_of_inferior
                K = len(selected_clients)
                coef = I / K
                global_dict[k] = global_dict[k] - coef * means_dict[k]

        means_dict = {k: v.mean().item() for k, v in global_dict.items()}
        print("After-shifting", means_dict)

        global_model.load_state_dict(global_dict)

        global_model.to(device)

        # Evaluate the global model after each communication round
        print("Evaluating global model:")
        evaluate(global_model, device, testloader, communication_round)
        torch.cuda.empty_cache()
    # if communication_round % 10 == 0:
    checkpoint_path = f"checkpoints/{run_name}"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save(
        global_model.state_dict(),
        f"checkpoints/{run_name}/{communication_round:04d}.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedShift Arguments")

    # Adding arguments for bits and use_fedprox
    parser.add_argument(
        "--bits", type=int, default=4, help="Number of bits for quantization"
    )
    parser.add_argument(
        "--num_clients", type=int, default=100, help="Number of clients"
    )
    parser.add_argument(
        "--selected_clients", type=int, default=10, help="Selected clients"
    )

    parser.add_argument("--dir_alpha", type=float, default=0, help="dirichlet alpha")

    parser.add_argument("--seed", type=int, default=42, help="Random seed value")
    parser.add_argument(
        "--data_type",
        dest="data_type",
        action="store",
        help="CIFAR10 or CIFAR100",
    )
    parser.add_argument(
        "--use_fedprox",
        dest="use_fedprox",
        action="store_true",
        help="Use FedProx algorithm (default: True)",
    )
    parser.add_argument(
        "--no_fedprox",
        dest="use_fedprox",
        action="store_false",
        help="Don't use FedProx algorithm (default: Use)",
    )
    parser.add_argument(
        "--use_scaffold",
        dest="use_scaffold",
        action="store_true",
        help="Use scaffold algorithm (default: True)",
    )
    parser.add_argument(
        "--no_scaffold",
        dest="use_scaffold",
        action="store_false",
        help="Don't use scaffold algorithm (default: Use)",
    )
    parser.add_argument(
        "--use_fedshift",
        dest="use_fedshift",
        action="store_true",
        help="Use fedshift algorithm (default: True)",
    )
    parser.add_argument(
        "--no_fedshift",
        dest="use_fedshift",
        action="store_false",
        help="Don't use fedshift algorithm (default: Use)",
    )
    parser.add_argument(
        "--use_uniform_q",
        dest="use_uniform_q",
        action="store_true",
        help="Use uniform quantization (default: True)",
    )
    parser.add_argument(
        "--no_uniform_q",
        dest="use_uniform_q",
        action="store_false",
        help="Use uniform quantization (default: True)",
    )

    parser.add_argument(
        "--use_quantization",
        dest="use_quantization",
        action="store_true",
        help="Use quantization (default: True)",
    )
    parser.add_argument(
        "--no_quantization",
        dest="use_quantization",
        action="store_false",
        help="Not using quantization (default: True)",
    )
    parser.set_defaults(
        use_fedprox=False,
        use_fedshift=False,
        use_uniform_q=False,
        use_scaffold=False,
        use_quantization=False,
    )

    args = parser.parse_args()
    main(args)
