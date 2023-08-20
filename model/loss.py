import torch
import numpy as np

categories = [
    "all-body",
    "bags",
    "tops",
    "outerwear",
    "hats",
    "bottoms",
    "scarves",
    "jewellery",
    "accessories",
    "shoes",
    "sunglasses"
]


# output: [batch_size, num_categories, d_model]
# embeddings: [batch_size, num_categories, d_model]
# target_mask: [batch_size, num_categories]
def contrastive(output, output_2, item_indices, embeddings, target_mask, target_cls, dataset):
    loss_1 = first_loss(output, embeddings, target_mask)
    loss_2 = second_loss(output, item_indices, embeddings, target_mask, dataset)
    loss_3 = torch.nn.functional.cross_entropy(output_2, target_cls)
    return loss_1 + loss_2 + loss_3


def first_loss(output, embeddings, target_mask):
    device = output.device
    loss_1 = torch.tensor(0.0).to(device)
    counter_1 = 0

    n = embeddings.shape[0]
    num_categories = embeddings.shape[1]

    empty = torch.zeros(n, n).to(device)

    for i in range(num_categories):
        _output = output[:, i, :].squeeze(1)
        _embeddings = embeddings[:, i, :].squeeze(1)
        _target_mask = target_mask[:, i].unsqueeze(-1).float()

        cos_sim = _embeddings @ _output.transpose(0, 1)
        exp_cos_sim = torch.exp(cos_sim)

        mask = (_target_mask @ _target_mask.transpose(0, 1)).bool()
        masked_exp_cos_sim = torch.where(mask, exp_cos_sim, empty)

        for j in range(n):
            if mask[j][j]:
                _numer = masked_exp_cos_sim[j][j]
                _denom_1 = torch.sum(masked_exp_cos_sim[j, :])
                loss_1 -= torch.log(_numer / _denom_1)
                counter_1 += 1

                # _denom_2 = torch.sum(masked_exp_cos_sim[:, j])
                # loss_1 -= torch.log(_numer / _denom_2)
                # counter_1 += 1

    return loss_1 / counter_1


def second_loss(output, item_indices, embeddings, target_mask, dataset):
    device = output.device
    loss_2 = torch.tensor(0.0).to(device)
    counter_2 = 0

    n = embeddings.shape[0]
    num_categories = embeddings.shape[1]

    for i in range(num_categories):
        cat_indices = dataset.index_categories[categories[i]]
        cat_embeddings = dataset.index_embeddings[cat_indices].to(device)

        _target_mask = target_mask[:, i].unsqueeze(-1).float()
        mask = (_target_mask @ _target_mask.transpose(0, 1)).bool()

        for j in range(n):
            if mask[j][j]:
                _output = output[j, i]

                _pos_index = item_indices[j, i].item()
                _pos_embedding = embeddings[j, i]
                _numer = torch.exp(_output @ _pos_embedding)
        
                temp_sim = torch.exp(_output @ cat_embeddings.transpose(0, 1)).squeeze()
                _neg_indices = torch.topk(temp_sim, n, largest = False).indices
                _neg_embeddings = cat_embeddings[_neg_indices].detach()
                _denom = torch.sum(torch.exp(_output @ _neg_embeddings.transpose(0, 1)))
                if _pos_index not in _neg_indices:
                    _denom += _numer

                loss_2 -= torch.log(_numer / _denom)
                counter_2 += 1
    
    return loss_2 / counter_2