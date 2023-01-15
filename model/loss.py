import torch.nn.functional as F

# output: [batch_size, num_categories, d_model]
# embeddings: [batch_size, num_categories, d_model]
# target_mask: [batch_size, num_categories]
def mse(output, embeddings, target_mask):
    mask = target_mask.unsqueeze(-1).float()
    y_hat = output * mask
    y = embeddings * mask
    return F.mse_loss(y_hat, y)