import numpy as np
import torch
from sklearn.cluster import KMeans as SklearnKMeans
from fast_pytorch_kmeans import KMeans as FastKMeans
import random


def quantization(model, BITS, is_uniform=True):
    if is_uniform:
        weight_dict = {}
        min_max_dict = {}
        BINS = 2**BITS
        for k, v in model.items():
            v = v.cpu().detach()
            mn = v.min()
            mx = v.max()
            v = ((v - mn) / (mx - mn)) * (BINS - 1)
            weight_dict[k] = torch.round(v).to(torch.uint8)
            min_max_dict[k] = (mn, mx)

        return weight_dict, min_max_dict

    else:
        weight_dict = {}
        codebook_dict = {}
        cuda_available = torch.cuda.is_available()
        cuda_available = True

        for k, v in model.items():
            original_shape = v.shape

            if not cuda_available and isinstance(v, torch.Tensor):
                v = v.cpu().numpy()

            x = v.reshape(-1).reshape(-1, 1)

            if not torch.cuda.is_available():
                x = x.cpu()

            if cuda_available:
                kmeans = FastKMeans(
                    n_clusters=int(2**BITS), max_iter=300, mode="euclidean", verbose=0
                )
                labels = kmeans.fit_predict(x.float())
            else:
                kmeans = SklearnKMeans(n_clusters=int(2**BITS), n_init="auto").fit(x)
                labels = kmeans.labels_
            weight_dict[k] = labels.reshape(original_shape)
            codebook_dict[k] = (
                kmeans.cluster_centers_ if not cuda_available else kmeans.centroids
            )
            if not cuda_available:
                weight_dict[k] = torch.from_numpy(weight_dict[k])

        return weight_dict, codebook_dict


def dequantization(I, BITS, extra_info, is_uniform=True):
    if is_uniform:
        BINS = 2**BITS
        min_v, max_v = extra_info

        result = np.linspace(min_v, max_v, BINS)[I]

        return torch.from_numpy(result)

    else:
        codebook = extra_info
        result = np.squeeze(codebook[I], axis=-1)
        return result
