import numpy as np
import torch
from torch.nn import functional as F


EPSILON = 1e-8

def sample_mi(oh_features, var_type='discrete'):
    """ Estimate the mutual information between all pairwise sets of features.
    
    Args:
        oh_features: torch.Tensor, shape (n_samples, n_features, feature_dim)
        var_type: str, 'discrete' or 'continuous'
    
    Returns:
        torch.Tensor, shape ()
    """
    assert var_type in ['discrete', 'continuous'], \
        f'var_type must be either discrete or continuous, not {var_type}!'

    if var_type == 'continuous':
        raise NotImplementedError('Continuous MI is not implemented yet!')

    feature_entropies = one_hot_entropy(oh_features.mean(dim=0))

    n_samples, n_features = oh_features.shape[:2]
    total_mutual_info = torch.tensor(0, dtype=torch.float32, device=oh_features.device)
    total_su = torch.tensor(0, dtype=torch.float32, device=oh_features.device)
    for i in range(n_features):
        for j in range(i+1, n_features):
            oh_xs = oh_features[:, i]
            oh_ys = oh_features[:, j]

            # H(x|y) = -sum_x[ sum_y[ p(x, y) log(p(x, y) / p(y)) ] ]
            joint_probs = torch.mm(oh_xs.T, oh_ys) / n_samples
            y_probs = torch.mean(oh_ys, dim=0).unsqueeze(0)
            conditional_entropy_mat = joint_probs * torch.log(
                (joint_probs / (y_probs + EPSILON)) + EPSILON)
            conditional_entropy = -torch.sum(conditional_entropy_mat)

            mutual_info = feature_entropies[i] - conditional_entropy # I(x, y) = H(x) - H(x|y)
            total_mutual_info += mutual_info
            total_su += 2 * mutual_info / (
                feature_entropies[i] + feature_entropies[j] + EPSILON)

    return total_mutual_info, total_su

class HashableTensorWrapper(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor, *args, hash=None, **kwargs):
        return super().__new__(cls, tensor, *args, **kwargs)

    def __init__(self, tensor, hash=None):
        self._tensor = tensor
        self._hash = hash

    # def __getattr__(self, attr):
    #     return getattr(self._tensor, attr)
    def __getitem__(self, i):
        return HashableTensorWrapper(self._tensor.__getitem__(i))
    def __hash__(self):
        if self._hash is None:
            # print('Warning: HashableTensorWrapper hash not set!')
            self._hash = hash_tensor(self._tensor).item()
        return self._hash
    def set_hash(self, hash):
        self._hash = hash
    def __repr__(self):
        return self._tensor.__repr__()
    def __str__(self):
        return self._tensor.__str__()
    def __len__(self):
        return self._tensor.__len__()
    def __eq__(self, other):
        return (self._tensor == other._tensor).all()

def as_hashable(x):
    if isinstance(x, torch.Tensor):
        return HashableTensorWrapper(x)
    raise NotImplementedError(f'Cannot convert {type(x)} to hashable!')

def hash_tensor(tensor, bins=int(1e6)):
    order_add = torch.linspace(0, 1, tensor.shape.numel(), device=tensor.device)
    order_add = order_add.reshape(tensor.shape)
    return torch.quantize_per_tensor(int(1e6) * (tensor + order_add), 0.1, 0, torch.qint32) \
        .int_repr().reshape(-1).sum() % bins

def hash_tensors(tensors, bins=int(1e6)):
    order_add = torch.linspace(0, 1, tensors[0].shape.numel(), device=tensors[0].device)
    order_add = order_add.reshape([1] + list(tensors[0].shape))
    return (torch.quantize_per_tensor(int(1e6) * (tensors + order_add), 0.1, 0, torch.qint32) \
        .int_repr().reshape(tensors.shape[0], -1).sum(-1) % bins).tolist()

def to_hashable_tensor_list(tensor):
    """ Convert a N-D tensor into a list of (N-1)-D hashable tensors. """
    hashes = hash_tensors(tensor)
    tensor_list = []
    for i in range(len(tensor)):
        tensor_list.append(HashableTensorWrapper(tensor[i], hash=hashes[i]))
    return tensor_list

def categorical_kl_div(pred, target):
    """ Calculate KL Divergence for categorical distributions. """
    return torch.sum(target * torch.log(target / (pred + EPSILON) + EPSILON), dim=-1)

def one_hot_cross_entropy(pred, target):
    """ Calculate the cross entropy between two one-hot vectors. """
    return -torch.sum(target * torch.log(pred + EPSILON), dim=-1)

def one_hot_entropy(probs):
    """ Calculate the cross entropy between two one-hot vectors. """
    # Input: (n_features, discrete_dim)
    return -torch.sum(probs * torch.log(probs + EPSILON), dim=-1)

# dist = Categorical(torch.tensor([0.6, 0.4, 0.0]))
# logits = torch.rand((1, dist.support.upper_bound + 1,), requires_grad=True)
# optimizer = torch.optim.Adam((logits,), lr=1e-2)
# print(F.softmax(logits, dim=1))
# for i in range(10000):
#     samples = dist.sample((1000,))
#     oh_samples = F.one_hot(samples, num_classes=dist.support.upper_bound + 1).float()
#     probs = F.softmax(logits, dim=1)
#     # loss = one_hot_cross_entropy(oh_samples, probs).mean()
#     loss = categorical_kl_div(probs, oh_samples).mean()
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if i % 1000 == 0:
#         print(F.softmax(logits, dim=1), loss.item())

# Reference: https://gnarlyware.com/blog/mutual-information-exaxmple-with-categorical-variables/
def sample_symmetric_uncertainty(features, var_type='discrete'):
    """ Estimate the symmetric uncertainty between all pairwise sets of features.
    
    Args:
        features: torch.Tensor, shape (n_samples, n_features)
        var_type: str, 'discrete' or 'continuous'
    
    Returns:
        torch.Tensor, shape (n_features, n_features)
    """
    assert var_type in ['discrete', 'continuous'], \
        f'var_type must be either discrete or continuous, not {var_type}!'

    if var_type == 'continuous':
        raise NotImplementedError('Continuous symmetric uncertainty is not implemented yet!')

    # Formula: 2 * I(x, y) / (H(x), H(y))
    # I(x, y) = H(x) - H(x|y) = H(y) - H(y|x) 
    discrete_dim = features.max() + 1
    one_hot_features = F.one_hot(features, num_classes=discrete_dim).float()
    feature_entropies = sample_entropy(features.transpose(0, 1), var_type=var_type)

    n_samples, n_features = features.shape
    sym_uncertainty = torch.eye(n_features, dtype=torch.float32, device=features.device)
    for i in range(features.shape[1]):
        for j in range(i+1, features.shape[1]):
            oh_xs = one_hot_features[:, i]
            oh_ys = one_hot_features[:, j]

            # H(x|y) = -sum_x[ sum_y[ p(x, y) log(p(x, y) / p(y)) ] ]
            joint_probs = torch.mm(oh_xs.T, oh_ys) / n_samples
            y_probs = torch.mean(oh_ys, dim=0).unsqueeze(0)
            conditional_entropy_mat = joint_probs * torch.log(
                (joint_probs / (y_probs + EPSILON)) + EPSILON)
            conditional_entropy = -torch.sum(conditional_entropy_mat)

            mutual_info = feature_entropies[i] - conditional_entropy # I(x, y) = H(x) - H(x|y)
            sym_uncertainty[i, j] = 2 * mutual_info / (feature_entropies[i] + feature_entropies[j])
            sym_uncertainty[j, i] = sym_uncertainty[i, j]

    return sym_uncertainty

def triu_avg(mat, nan_replacement=0):
    """ Average the upper triangular part of a matrix.
    
    Args:
        mat: torch.Tensor, square matrix
    """
    return mat.triu(1).nan_to_num(nan_replacement).sum() / \
        (np.prod(mat.shape) - mat.shape[0]) * 2

def sample_entropy(xs, var_type='discrete'):
    """ Estimate the entropy of the samples from random variable X.
    
    Args:
        xs: torch.Tensor, shape (n_features, n_samples)
    """
    assert var_type in ['discrete', 'continuous'], \
        f'var_type must be either discrete or continuous, not {var_type}!'

    if var_type == 'continuous':
        raise NotImplementedError('Continuous entropy is not implemented yet!')

    discrete_dim = xs.max() + 1
    # (n_features, n_smaples) -> (n_features, n_samples, discrete_dim)
    one_hots = F.one_hot(xs, num_classes=discrete_dim).float()
    # (n_features, n_samples, discrete_dim) -> (n_features, discrete_dim)
    probs = one_hots.mean(dim=1)
    # (n_features, discrete_dim) -> (n_features,)
    entropy = -torch.sum(probs * torch.log(probs + EPSILON), dim=1)
    return entropy