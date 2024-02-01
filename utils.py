import random
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as F


def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    # Python
    random.seed(seed_value)

    # Numpy
    np.random.seed(seed_value)

    # PyTorch
    torch.manual_seed(seed_value)

    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(
                    f"[ERROR] ...initialization method [{init_type}] is not implemented!"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif (
            classname.find("BatchNorm2d") != -1
            or classname.find("InstanceNorm2d") != -1
        ):
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def client_selection(num_selected_clients, num_clients, mode="all"):
    selected_clients = random.sample(range(num_clients), num_selected_clients)
    if mode == "superior":
        even_numbers = [i for i in range(num_clients) if i % 2 == 0]
        selected_clients = random.sample(even_numbers, num_selected_clients)
    elif mode == "inferior":
        even_numbers = [i for i in range(num_clients) if i % 2 == 1]
        selected_clients = random.sample(even_numbers, num_selected_clients)
    elif mode == "half-odd-even":
        # Assuming num_selected_clients is always an even number
        half = num_selected_clients // 2

        # Generate a list of client IDs
        client_ids = list(range(0, num_clients))

        # Shuffle the client_ids to ensure randomness
        random.shuffle(client_ids)

        # Select half odd and half even clients
        odd_clients = [client for client in client_ids if client % 2 != 0][:half]
        even_clients = [client for client in client_ids if client % 2 == 0][:half]

        # Combine the two lists
        selected_clients = odd_clients + even_clients

        # Shuffle the combined list to avoid any order bias
        random.shuffle(selected_clients)
    return selected_clients


# Function to calculate L2 distance between two sets of model parameters
def l2_distance(global_model, prev_global_model):
    l2_dist = {
        k: torch.norm(prev_global_model[k].to("cpu") - global_model[k].to("cpu")).item()
        for k in global_model
    }
    return l2_dist


# Function to calculate cosine similarity between two sets of model parameters
def cosine_similarity(global_model, prev_global_model):
    cos_sim = {}
    for k in global_model:
        # Flatten the tensors to 1-D
        global_tensor = global_model[k].view(-1).to("cpu")
        prev_global_tensor = prev_global_model[k].view(-1).to("cpu")

        # Compute cosine similarity over 1-D tensors
        cos_sim[k] = F.cosine_similarity(
            prev_global_tensor.unsqueeze(0), global_tensor.unsqueeze(0), dim=1
        ).item()
    return cos_sim


# Function to calculate KL divergence between two sets of model parameters
# This is not a standard use of KL divergence; here we are treating model weights as probability distributions
def compute_KL_divergence(model1, model2):
    kl_divergences = {}
    for (name1, param1), (name2, param2) in zip(model1.items(), model2.items()):
        # Verify that the parameter names match
        if name1 != name2:
            raise ValueError(f"Parameter names do not match: {name1} and {name2}")

        param1 = param1.detach().to("cpu").flatten()
        param2 = param2.detach().to("cpu").flatten()

        # Avoid division by zero in case std() returns 0.
        std1 = param1.std().clamp(min=1e-12)
        std2 = param2.std().clamp(min=1e-12)

        # Representing the parameters as normal distributions
        dist1 = torch.distributions.Normal(param1, std1)
        dist2 = torch.distributions.Normal(param2, std2)

        # Calculating the KL divergence for the current parameter
        kl_div = torch.distributions.kl_divergence(dist1, dist2).sum().item()
        kl_divergences[name1] = kl_div

    return kl_divergences


# Function to calculate covariance between two sets of model parameters
def covariance(global_model, prev_global_model):
    cov = {}
    for k in global_model:
        x = prev_global_model[k].to("cpu").flatten()
        y = global_model[k].to("cpu").flatten()
        cov[k] = torch.cov(torch.stack((x, y)))[0, 1].item()
    return cov


# # Function to log all the distances using wandb
# def log_distances(
#     distance_name,
#     distances_shift,
#     distances_non_shift,
#     communication_round,
# ):
#     wandb.log(
#         {f"{distance_name}-Shift-{k}": distances_shift[k] for k in distances_shift},
#         step=communication_round,
#     )
#     wandb.log(
#         {
#             f"{distance_name}-Non-Shift-{k}": distances_non_shift[k]
#             for k in distances_non_shift
#         },
#         step=communication_round,
#     )
