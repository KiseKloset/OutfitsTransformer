import torch
import numpy as np


def recall_at5(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 5)


def recall_at10(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 10)


'''
index_names: DataLoader.index_names
sorted_index_names: Dataset.query_top_items
target_names: item_indexes[target_mask]
'''
def recall_at_k(index_names, sorted_index_names, target_names, k = 1):
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    recall = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return recall