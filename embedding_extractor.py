"""
Embedding extractor
"""

import torch
import numpy as np
import pickle
from typing import Dict

class EmbeddingExtractor :
    def __init__(self, model, ent2id: Dict, rel2id: Dict, device = "cpu") :
        self.model = model
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = {v: k for k, v in ent2id.items()}
        self.id2rel = {v: k for k, v in rel2id.items()}
        self.device = device

        # Now extract embeddings
        self.entity_embeddings = None
        self.relation_embeddings = None
        self._extract_embeddings()

    def _extract_embeddings(self):
        """func to get embeddings"""
        self.model.eval()
        with torch.no_grad() :
            self.entity_embeddings = self.model.entity_emb.weight.cpu().numpy()
            self.relation_embeddings = self.model.relation_emb.weight.cpu().numpy()

        print(f"Extracted ent_emb : {self.entity_embeddings.shape}")
        print(f"extracted rel_emb : {self.relation_embeddings.shape}")

    def get_entity_embedding(self, entity: str) -> np.ndarray :
        """Get embedding for specific entity"""
        if entity not in self.ent2id:
            raise ValueError(f"Entity '{entity}' non found in vocabulary")
        entity_id = self.ent2id[entity]
        return self.entity_embeddings[entity_id]
    
    def get_relation_embedding(self, relation: str) -> np.ndarray:
        """Get embedding for a specific relation"""
        if relation not in self.rel2id :
            raise ValueError(f"Relation '{relation}' not found in vocaubaly")
        relation_id = self.rel2id[relation]
        return self.relation_embeddings[relation_id]
    
    def get_all_entity_embeddings(self) -> np.ndarray :
        """Get all entity embeddings"""
        return self.entity_embeddings
    
    def get_all_relation_embeddings(self) -> np.ndarray :
        """Get all relation entity embeddings"""
        return self.relation_embeddings
    
    def save_embeddings(self, filepath: str) :
        """Save embeddings to file"""
        data = {
           "entity_embeddings" : self.entity_embeddings,
            "relation_embeddings" : self.relation_embeddings,
            "ent2id" : self.ent2id,
            "rel2id" : self.rel2id,
            "id2ent" : self.id2ent,
            "id2rel" : self.id2rel
        }

        with open(filepath, "wb") as f :
            pickle.dump(data, f)
        print(f"embeddings save to {filepath}")

    @classmethod
    def load_embeddings(cls, filepath: str) :
        """Load embeddings from file"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Ebeddings loaded from {filepath}")
        return data