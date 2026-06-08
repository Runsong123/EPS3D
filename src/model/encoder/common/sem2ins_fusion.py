import torch
import torch.nn as nn


class Sem2InsFusion(nn.Module):
    """Semantic-aware Instance (Sem2Ins) feature fusion module.

    I_g = I_g + w_1 * F_fusion(concat(F_proj1(I_g), F_proj2(S_g)))
    """

    def __init__(self, instance_dim=32, semantic_dim=32, hidden_dim=32, w_init=0.1):
        super().__init__()
        self.proj_instance = nn.Linear(instance_dim, hidden_dim)
        self.proj_semantic = nn.Linear(semantic_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, instance_dim),
        )
        self.w_1 = nn.Parameter(torch.tensor(w_init))


    def forward(self, instance_feat, semantic_feat):
        # DPT head outputs (B, V, C, H, W); move channel dim to last for nn.Linear
        needs_permute = instance_feat.dim() >= 4
        if needs_permute:
            inst = instance_feat.movedim(2, -1)
            sem = semantic_feat.movedim(2, -1)
        else:
            inst = instance_feat
            sem = semantic_feat

        proj_i = self.proj_instance(inst)
        proj_s = self.proj_semantic(sem)
        fused = self.fusion(torch.cat([proj_i, proj_s], dim=-1))

        if needs_permute:
            fused = fused.movedim(-1, 2)

        return instance_feat + self.w_1 * fused
