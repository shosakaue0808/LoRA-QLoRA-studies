import torch
import torch.nn as nn

class LayerWithLoRA(nn.Module):
    """
    represents updated layer with LoRA layer attached to a given base layer
    """
    def __init__(self, base_layer, rank, alpha):
        super().__init__()
        self.base = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        # set base weights freeze just in case
        for param in self.base.parameters():
            param.requires_grad = False

        weight = base_layer.weight
        device = weight.device
        dtype = torch.bfloat16
        self.A = nn.Parameter(torch.randn(base_layer.in_features, rank, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, base_layer.out_features, device=device, dtype=dtype))
        #DW = AB = 0 for initialization
        self.alpha = alpha # scale ABx by alpha/rank usually 1:1 or 1:2
        self.rank = rank

    def forward(self, x):
        base_out = self.base(x)
        x_lora = x.to(self.A.dtype)
        lora_out = (self.alpha/self.rank) * (x_lora @ self.A @ self.B)
        lora_out = lora_out.to(base_out.dtype)
        return base_out + lora_out

def attach_Lora_to_Linear(base_model, rank, alpha):
    base_model.rank = rank
    base_model.alpha = alpha

    for layer in base_model.layers:
        layer.self_attn.q_proj = LayerWithLoRA(layer.self_attn.q_proj, rank=rank, alpha=alpha)
        layer.self_attn.k_proj = LayerWithLoRA(layer.self_attn.k_proj, rank=rank, alpha=alpha)
        layer.self_attn.v_proj = LayerWithLoRA(layer.self_attn.v_proj, rank=rank, alpha=alpha)
        layer.self_attn.o_proj = LayerWithLoRA(layer.self_attn.o_proj, rank=rank, alpha=alpha)
        layer.mlp.gate_proj = LayerWithLoRA(layer.mlp.gate_proj, rank=rank, alpha=alpha)
        layer.mlp.down_proj = LayerWithLoRA(layer.mlp.down_proj, rank=rank, alpha=alpha)
        layer.mlp.up_proj = LayerWithLoRA(layer.mlp.up_proj, rank=rank, alpha=alpha)
