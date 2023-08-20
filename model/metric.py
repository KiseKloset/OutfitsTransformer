import torch
import numpy as np


def recall_at1(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 1)

def recall_at2(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 2)

def recall_at3(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 3)

def recall_at5(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 5)

def recall_at10(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 10)

def recall_at30(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 30)

def recall_at50(index_names, sorted_index_names, target_names):
    return recall_at_k(index_names, sorted_index_names, target_names, 50)


'''
predicted_target_indices: [batch, num_cat, 10] # indices
items_indices: [batch, num_cat]
target_mask: [batch, num_cat]
'''
def recall_at_k(predicted_target_indices, item_indices, target_mask, k = 1):
    _item_indices = item_indices.unsqueeze(-1)
    _target_mask = target_mask.unsqueeze(-1)
    
    labels = predicted_target_indices == _item_indices

    all_cases = torch.sum(_target_mask).item()
    correct_cases = torch.sum((labels & _target_mask)[..., :k]).item()

    recall = correct_cases / all_cases * 100.0
    return recall