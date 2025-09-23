import torch
import torch.nn as nn


class ModalitiesAttentionAdd(nn.Module):
    def __init__(self,
                 modality_count: int,
                 fused_size: int,
                 masked_weight=-1e5,
                 temperature=1.0):
        super(ModalitiesAttentionAdd, self).__init__()
        self.modality_count = modality_count
        self.fused_size = fused_size
        self.masked_weight = masked_weight
        self.temperature = temperature

        init_attention = torch.zeros(modality_count, fused_size, dtype=torch.float32)
        self.attention_weights = nn.Parameter(init_attention)

    def forward(self, x, mask=None) -> torch.Tensor:
        if mask is not None:
            # Mask shape: [batch_size, modality_count]
            # Attention new shape: [1, modality_count, fused_size]
            attention_weights = self.attention_weights.unsqueeze(dim=0)
            # Mask new shape: [batch_size, modality_count, 1]
            mask = mask.unsqueeze(dim=-1)
            # Attention new shape: [batch_size, modality_count, fused_size]
            attention_weights += torch.where(mask, 0.0, self.masked_weight)
            attention_weights = torch.softmax(attention_weights * self.temperature, dim=1)

            return torch.einsum("bns,bns->bs", x, attention_weights)

        else:
            # Attention shape: [modality_count, fused_size]
            # Input shape: [batch_size, modality_count, fused_size]
            attention_weights = torch.softmax(self.attention_weights * self.temperature, dim=0)
            return torch.einsum("bns,ns->bs", x, attention_weights)
