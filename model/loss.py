import torch.nn as nn

# output: [batch_size, num_categories, d_model]
# embeddings: [batch_size, num_categories, d_model]
# target_mask: [batch_size, num_categories]
def mse(output, embeddings, target_mask):
    mask = target_mask[:, :, None].float()
    y_hat = output * mask
    y = embeddings * mask
    return nn.MSELoss(y_hat, y)
    
