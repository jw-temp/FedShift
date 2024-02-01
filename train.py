import copy
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from quantization import quantization
from tqdm import tqdm
from utils import set_seed


import time


def proximal_term(local_model, global_model_state_dict, mu=0.01):
    """Compute the proximal regularization term."""
    reg_loss = 0.0
    for param, global_param in zip(
        local_model.parameters(), global_model_state_dict.values()
    ):
        reg_loss += torch.norm(param - global_param) ** 2
    return mu / 2.0 * reg_loss


def client_training(
    client_id,
    model,
    device,
    train_loader,
    epochs=1,
    bits=4,
    use_fedprox=True,
    is_uniform_q=True,
    use_quantization=True,
    mu=0.01,
    seed=42,
    use_scaffold=False,
    global_control_variate=None,
    local_control_variate=None,
):
    set_seed(seed)
    # Get the unique labels in the client's dataset
    all_labels = []
    for _, (_, target) in enumerate(train_loader):
        all_labels.extend(target.tolist())
    unique_labels = set(all_labels)

    # Print client details
    print(
        f"Client {client_id} started training, dataset size: {len(train_loader.dataset)}. Contains labels: {unique_labels}"
    )

    criterion = nn.CrossEntropyLoss()
    lr = 0.005
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.to(device)
    global_model_state_dict = copy.deepcopy(model.state_dict())
    global_model = copy.deepcopy(model)
    model.train()

    # Compute control variate difference (c_diff)
    c_diff = {
        k: global_control_variate[k] - local_control_variate[k]
        for k in local_control_variate.keys()
    }

    # Training

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # Incorporate proximal term if using FedProx
            if use_fedprox:
                loss += proximal_term(model, global_model_state_dict, mu)

            loss.backward()

            if use_scaffold:
                # Apply the control variate difference to the gradient
                for name, param in model.named_parameters():
                    if param.grad is not None and name in c_diff:
                        c_d = c_diff[name]

                        # Move to the device only if it's not already there
                        if c_d.device != param.device:
                            c_d = c_d.to(param.device)
                        param.grad += c_d

            optimizer.step()
    # Update local control variate (c_local)

    c_plus = {}
    if use_scaffold:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in local_control_variate and name in global_control_variate:
                    global_param = global_model.state_dict()[name]
                    c_l = local_control_variate[name]
                    c_g = global_control_variate[name]

                    # Move to the device only if not already there
                    if c_l.device != param.device:
                        c_l = c_l.to(param.device)
                    if c_g.device != param.device:
                        c_g = c_g.to(param.device)
                    if global_param.device != param.device:
                        global_param = global_param.to(param.device)

                    coef = 1.0 / (len(train_loader.dataset) * epochs * lr)
                    c_plus_i = c_l - c_g + coef * (global_param - param)
                    c_plus[name] = c_plus_i.to("cpu")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Client {client_id} finished training. Accuracy on its data: {accuracy:.2f}%"
    )
    if client_id % 2 == 1 and use_quantization:
        start = time.time()
        q_state_dict, extra_info = quantization(model.state_dict(), bits, is_uniform_q)
        end = time.time()
        print("Done", end - start)
        device_state_dict = {k: v.to(device) for k, v in q_state_dict.items()}
        model.load_state_dict(device_state_dict)

    else:
        extra_info = None

    model.to("cpu")
    return model.state_dict(), extra_info, c_plus


def client_training_process(args):
    (
        client_id,
        model,
        device,
        train_loader,
        epochs,
        bits,
        use_fedprox,
        is_uniform_q,
        use_quantization,
        mu,
        seed,
        use_scaffold,
        global_control_variate,
        local_control_variate,
    ) = args
    state_dict, codebook, c_plus = client_training(
        client_id,
        model,
        device,
        train_loader,
        epochs,
        bits,
        use_fedprox,
        is_uniform_q,
        use_quantization,
        mu,
        seed,
        use_scaffold,
        global_control_variate,
        local_control_variate,
    )

    return client_id, (state_dict, codebook), c_plus


def evaluate(model, device, test_loader, round, is_shifting=True):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct_topN = 0
    top_n = 1
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            # Get top N predictions
            _, topN_pred = output.topk(top_n, dim=1)
            correct_topN += (
                topN_pred.eq(target.view(-1, 1).expand_as(topN_pred)).sum().item()
            )

    test_loss /= len(test_loader.dataset)

    top3_accuracy = 100.0 * correct_topN / len(test_loader.dataset)

    print(
        f"Test set: Average loss: {test_loss:.4f}, Top-N Accuracy: {correct_topN}/{len(test_loader.dataset)} ({top3_accuracy:.0f}%)"
    )
    if math.isnan(test_loss):
        print("Test loss is NaN. Exiting...")
        return
