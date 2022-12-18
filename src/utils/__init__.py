from munch import Munch
import torch

config = None

def get_config():
    global config
    if config is None:
        with open('./src/config.yaml', 'rt', encoding='utf-8') as f:
            config = Munch.fromYAML(f.read())
    return config

def apply_padding(tensors_list, padding_value=0):
    """Applies padding to a list of tensors.

    Args:
        tensors_list (list): list of tensors to pad
        padding_value (int): value to use for padding

    Returns:
        tuple: tuple containing:
            - **padded_tensors** (torch.Tensor): tensor of shape (batch_size, max_seq_len, ...) containing the padded tensors
            - **tensors_lengths** (torch.Tensor): tensor of shape (batch_size, ) containing the length of each tensor"""

    tensors_lengths = torch.tensor([t.shape[1] for t in tensors_list], dtype=torch.int64)
    max_len = tensors_lengths.max()
    padded_tensors = [torch.nn.functional.pad(t, pad=(0, 0, 0, max_len - t.shape[1]), value=padding_value) for t in tensors_list]

    return torch.cat(padded_tensors, dim=0), tensors_lengths
