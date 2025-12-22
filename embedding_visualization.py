"""This module controls the visualization pipeline"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import embedding_extractor
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class EmbeddingVisualizer :
    def __init__(self, extractor: embedding_extractor.EmbeddingExtractor) :
        self.extractor = extractor 

    def reduce_dimensions_tsne(self, embeddings: np.ndarray,
                            perplexity : int = 30,
                            n_components : int = 2,
                            random_state : int = 42) -> np.ndarray :
        """Apply t-sne for dimensionality reduction"""
        print(f"applying t-sne with perplexity : {perplexity}")

        max_perplexity = min(30, embeddings.shape[0] - 1)
        perplexity = min(perplexity, max_perplexity)

        tsne = TSNE(n_components = n_components,
                    perplexity = perplexity,
                    random_state = random_state,
                    max_iter = 1000)
        reduced = tsne.fit_transform(embeddings)
        print(f"t-sne completed: {reduced.shape}")
        return reduced
    
    def reduced_dimension_pca(self, embeddings: np.ndarray, 
                             n_components : int = 2) -> np.ndarray :
        """PCA"""
        print("Applying pca")
        pca = PCA(n_components = n_components, random_state = 42)
        reduced = pca.fit_transform(embeddings)
        print(f"pca completed : {reduced.shape}")
        print(f"Explained variance ratio : {pca.explained_variance_ratio_}")
        return reduced
    
    def plot_entity_embeddings(self, method = "tsne", 
                               highlight_entities : Optional[List[str]] = None,
                               figsize = (12, 8)) :
        """Plot entity embeddings in 2D"""
        embeddings = self.extractor.get_all_entity_embeddings()
        print(embeddings)

        # reduce dimensions
        if method == "tsne" :
            reduced = self.reduce_dimensions_tsne(embeddings)
        elif method == "pca" :
            reduced = self.reduced_dimension_pca(embeddings)
        else: raise ValueError(f"Unknown method : {method}")

        plt.figure(figsize = figsize)
        plt.scatter(reduced[ :, 0], reduced[ :, 1], alpha = 0.6, s = 50)

        if highlight_entities :
            for entity in highlight_entities :
                if entity in self.extractor.ent2id :
                    idx = self.extractor.ent2id[entity]
                    plt.scatter(reduced[idx, 0], reduced[idx, 1],
                                color = "red", s = 200, marker = "*",
                                edgecolors = "black", linewidths = 2)
                    plt.annotate(entity, (reduced[idx, 0], reduced[idx, 1]),
                                 fontsize = 10, fontweight = "bold")
                    
        plt.title(f"Entity Embeddings Visualization ({method.upper()})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha = 0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_relation_embeddings(self, method = "pca", figsize = (10, 6)) :
        """Plot relation embeddings in 2D"""
        embeddings = self.extractor.get_all_relation_embeddings()

        # Reduce dimensions
        if method == 'tsne':
            reduced = self.reduce_dimensions_tsne(embeddings, perplexity=5)
        elif method == 'pca':
            reduced = self.reduce_dimensions_pca(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        plt.figure(figsize=figsize)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=100, c='orange')
        
        # Annotate relations
        for i, rel_id in enumerate(self.extractor.id2rel.keys()):
            relation = self.extractor.id2rel[rel_id]
            plt.annotate(relation, (reduced[i, 0], reduced[i, 1]),
                        fontsize=9, alpha=0.8)
        
        plt.title(f'Relation Embeddings Visualization ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    

class KnowledgeGraphQuery :
    """Query system for trained KGE model"""

    def __init__(self, model, extractor : embedding_extractor.EmbeddingExtractor,
                 train_triplets: List[Tuple], device = "cpu") :
        self.model = model
        self.extractor = extractor
        self.device = device

        # Build KGE structure
        self.kg = defaultdict(lambda: defaultdict(set))
        self._build_kg_structure(train_triplets)

    def _build_kg_structure(self, triplets: List[Tuple]) :
        """Build kg structure for efficient querying"""
        for h, r, t in triplets:
            h_name = self.extractor.id2ent[h]
            r_name = self.extractor.id2rel[r]
            print(r_name)
            t_name = self.extractor.id2ent[t]

            self.kg[h_name][r_name].add(t_name)

        print(f"KG structure build: {len(self.kg)} head entities")

    def find_similar_entities(self, entity: str, 
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar entities using cosine similarity"""
        query_emb = self.extractor.get_entity_embedding(entity)
        all_embs = self.extractor.get_all_entity_embeddings()
        
        # Compute cosine similarity
        query_norm = query_emb / np.linalg.norm(query_emb)
        all_norms = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
        similarities = np.dot(all_norms, query_norm)
        
        # Get top-k (excluding query entity itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            entity_name = self.extractor.id2ent[idx]
            similarity = similarities[idx]
            results.append((entity_name, float(similarity)))
        
        return results
    

    def predict_tail(self, head: str, relation: str, 
                    top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict tail entities given (head, relation)"""
        if head not in self.extractor.ent2id:
            raise ValueError(f"Entity '{head}' not found")
        if relation not in self.extractor.rel2id:
            raise ValueError(f"Relation '{relation}' not found")
        
        h_id = self.extractor.ent2id[head]
        r_id = self.extractor.rel2id[relation]
        
        # Compute scores for all possible tails
        self.model.eval()
        with torch.no_grad():
            h_tensor = torch.tensor([h_id], device=self.device)
            r_tensor = torch.tensor([r_id], device=self.device)
            
            scores = []
            for t_id in range(len(self.extractor.id2ent)):
                t_tensor = torch.tensor([t_id], device=self.device)
                score = self.model.score(h_tensor, r_tensor, t_tensor)
                scores.append(score.item())
        
        # Get top-k predictions
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            entity_name = self.extractor.id2ent[idx]
            score = scores[idx]
            results.append((entity_name, float(score)))
        
        return results
    

    def analogy_query(self, entity_a: str, relation: str, 
                     entity_b: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Analogy query: A:relation::B:?
        Find entity C such that (A, relation, B) ~ (A, relation, C)
        """
        # Get embeddings
        emb_a = self.extractor.get_entity_embedding(entity_a)
        emb_b = self.extractor.get_entity_embedding(entity_b)
        rel_emb = self.extractor.get_relation_embedding(relation)
        
        # Compute target: b - a + a = b (in TransE: h + r ≈ t)
        # We want to find entities similar to this pattern
        target = emb_a + rel_emb
        
        all_embs = self.extractor.get_all_entity_embeddings()
        
        # Compute distances
        distances = np.linalg.norm(all_embs - target, axis=1)
        
        # Get top-k closest entities
        top_indices = np.argsort(distances)[:top_k+2]
        
        results = []
        for idx in top_indices:
            entity_name = self.extractor.id2ent[idx]
            # Skip the query entities
            if entity_name != entity_a and entity_name != entity_b:
                distance = distances[idx]
                results.append((entity_name, float(-distance)))  # Negative for sorting
                if len(results) == top_k:
                    break
        
        return results



class EmbeddingClassifier:
    """Train classifiers on entity embeddings"""
    
    def __init__(self, extractor: embedding_extractor.EmbeddingExtractor):
        self.extractor = extractor
        self.classifiers = {}
        
    def prepare_classification_data(self, 
                                   entity_labels: Dict[str, str]) -> Tuple:
        """
        Prepare data for classification
        entity_labels: dict mapping entity names to their labels/categories
        """
        X = []
        y = []
        entity_names = []
        
        for entity, label in entity_labels.items():
            if entity in self.extractor.ent2id:
                embedding = self.extractor.get_entity_embedding(entity)
                X.append(embedding)
                y.append(label)
                entity_names.append(entity)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Prepared classification data: {X.shape[0]} samples")
        print(f"Unique labels: {np.unique(y)}")
        
        return X, y, entity_names
    
    def train_classifier(self, X, y, classifier_name='logistic',
                        test_size=0.2, random_state=42):
        """Train a classifier on embeddings"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Train classifier
        if classifier_name == 'logistic':
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        
        print(f"Training {classifier_name} classifier...")
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Detailed report
        y_pred = clf.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.classifiers[classifier_name] = clf
        
        return clf, train_score, test_score
    
    def predict_entity_category(self, entity: str, 
                               classifier_name='logistic') -> str:
        """Predict category for a given entity"""
        if classifier_name not in self.classifiers:
            raise ValueError(f"Classifier '{classifier_name}' not trained")
        
        embedding = self.extractor.get_entity_embedding(entity)
        prediction = self.classifiers[classifier_name].predict([embedding])[0]
        
        return prediction




class EmbeddingAnalyzer:
    """Analyze embedding quality and properties"""
    
    def __init__(self, extractor: embedding_extractor.EmbeddingExtractor):
        self.extractor = extractor
    
    def compute_embedding_statistics(self) -> Dict:
        """Compute basic statistics of embeddings"""
        entity_embs = self.extractor.get_all_entity_embeddings()
        relation_embs = self.extractor.get_all_relation_embeddings()
        
        stats = {
            'entity_mean': np.mean(entity_embs),
            'entity_std': np.std(entity_embs),
            'entity_min': np.min(entity_embs),
            'entity_max': np.max(entity_embs),
            'relation_mean': np.mean(relation_embs),
            'relation_std': np.std(relation_embs),
            'relation_min': np.min(relation_embs),
            'relation_max': np.max(relation_embs),
        }
        
        print("Embedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        return stats
    
    def cluster_entities(self, n_clusters: int = 5) -> Tuple:
        """Cluster entities using K-means"""
        embeddings = self.extractor.get_all_entity_embeddings()
        
        print(f"Clustering entities into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute silhouette score
        silhouette = silhouette_score(embeddings, labels)
        print(f"Silhouette score: {silhouette:.4f}")
        
        # Organize entities by cluster
        clusters = defaultdict(list)
        for entity_id, label in enumerate(labels):
            entity_name = self.extractor.id2ent[entity_id]
            clusters[label].append(entity_name)
        
        print("\nCluster sizes:")
        for cluster_id, entities in clusters.items():
            print(f"  Cluster {cluster_id}: {len(entities)} entities")
        
        return labels, clusters, silhouette
    
    def analyze_embedding_norms(self):
        """Analyze norms of entity and relation embeddings"""
        entity_embs = self.extractor.get_all_entity_embeddings()
        relation_embs = self.extractor.get_all_relation_embeddings()
        
        entity_norms = np.linalg.norm(entity_embs, axis=1)
        relation_norms = np.linalg.norm(relation_embs, axis=1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(entity_norms, bins=30, alpha=0.7, color='blue')
        plt.xlabel('Norm')
        plt.ylabel('Frequency')
        plt.title('Entity Embedding Norms')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(relation_norms, bins=20, alpha=0.7, color='orange')
        plt.xlabel('Norm')
        plt.ylabel('Frequency')
        plt.title('Relation Embedding Norms')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        print(f"Entity norms - Mean: {np.mean(entity_norms):.4f}, Std: {np.std(entity_norms):.4f}")
        print(f"Relation norms - Mean: {np.mean(relation_norms):.4f}, Std: {np.std(relation_norms):.4f}")
        
        return plt.gcf()












train_triplets = [
    ('Alice', 'likes', 'Bob'),
    ('Alice', 'likes', 'Music'),
    ('Bob',  'likes', 'Music'),
    ('Bob',  'knows', 'Alice')
]

class KG:
    def __init__(self, triples):
        self.kg = defaultdict(lambda: defaultdict(set))
        print(self.kg)
        self._build_kg_structure(triples)

    def _build_kg_structure(self, triples):
        for h, r, t in triples:
            self.kg[h][r].add(t)

kg = KG(train_triplets)
print(kg.kg)

# demo lookups
print(kg.kg['Alice']['likes'])   # {'Bob', 'Music'}
print(kg.kg['Bob']['knows'])     # {'Alice'}
print(kg.kg['Alice']['knows'])





