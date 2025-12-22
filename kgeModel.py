import pandas as pd
import numpy as np
import torch
from torch import nn


class BaseKGEModel(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, device):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        
        # nn.init.xavier_uniform_(self.entity_emb.weight.data)
        # nn.init.xavier_uniform_(self.relation_emb.weight.data)
        
        self.device = device
        self.to(device)

    def get_embeddings(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)
        return h, r, t


class TransE(BaseKGEModel):
    def __init__(self, num_entities, num_relations, emb_dim, gamma=12.0, p_norm=1, device="cpu"):
        super().__init__(num_entities, num_relations, emb_dim, device)
        self.gamma = torch.tensor(gamma, device=device)
        self.p_norm = p_norm

    def score(self, h_idx, r_idx, t_idx):
        h, r, t = self.get_embeddings(h_idx, r_idx, t_idx)
        x = h + r - t
        print(f" x is :{x}")
        dist = torch.norm(x, p=self.p_norm, dim=-1)
        print(f"dist is : {dist}")
        return self.gamma - dist
    
class DistMult(BaseKGEModel):
    def __init__(self, num_entities, num_relations, emb_dim, device="cpu"):
        super().__init__(num_entities, num_relations, emb_dim, device)

    def score(self, h_idx, r_idx, t_idx):
        h, r, t = self.get_embeddings(h_idx, r_idx, t_idx)
        return torch.sum(h * r * t, dim=-1)

class ComplEx(BaseKGEModel):
    def __init__(self, num_entities, num_relations, emb_dim, device="cpu"):
        assert emb_dim % 2 == 0, "ComplEx embedding dim must be even"
        super().__init__(num_entities, num_relations, emb_dim, device)
        self.real_dim = emb_dim // 2

    def _split(self, x):
        return x[..., :self.real_dim], x[..., self.real_dim:]

    def score(self, h_idx, r_idx, t_idx):
        h, r, t = self.get_embeddings(h_idx, r_idx, t_idx)
        h_r, h_i = self._split(h)
        r_r, r_i = self._split(r)
        t_r, t_i = self._split(t)
        
        term1 = h_r * r_r * t_r
        term2 = h_r * r_i * t_i
        term3 = h_i * r_r * t_i
        term4 = -h_i * r_i * t_r
        
        score = term1 + term2 + term3 + term4
        return torch.sum(score, dim=-1)


class RotatE(BaseKGEModel):
    def __init__(self, num_entities, num_relations, emb_dim, gamma=12.0, device="cpu"):
        assert emb_dim % 2 == 0, "RotatE embedding dim must be even"
        super().__init__(num_entities, num_relations, emb_dim, device)
        self.real_dim = emb_dim // 2
        self.gamma = torch.tensor(gamma, device=device)
        self.pi = 3.141592653589793

    def _split(self, x):
        return x[..., :self.real_dim], x[..., self.real_dim:]

    def score(self, h_idx, r_idx, t_idx):
        h, r, t = self.get_embeddings(h_idx, r_idx, t_idx)
        h_r, h_i = self._split(h)
        t_r, t_i = self._split(t)
        
        phase = r / (self.entity_emb.weight.data.norm(p=2) + 1e-9)
        phase = phase[..., :self.real_dim]
        phase = phase * self.pi
        
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)
        
        rot_r = h_r * r_re - h_i * r_im
        rot_i = h_r * r_im + h_i * r_re
        
        re_diff = rot_r - t_r
        im_diff = rot_i - t_i
        dist = torch.sqrt(re_diff**2 + im_diff**2 + 1e-9).sum(dim=-1)
        
        return self.gamma - dist
    