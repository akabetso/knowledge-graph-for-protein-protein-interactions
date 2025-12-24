import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer
import os

import dataset
import kgeModel
import embedding_visualization 
import embedding_extractor
import importlib
importlib.reload(dataset)
importlib.reload(embedding_visualization)
importlib.reload(embedding_extractor)
print(dir(dataset))
print(dir(embedding_visualization))

# set device agnostic setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else: device = torch.device("cpu")
print(f"device : {device}")

# Get train val and test dataset.
#train, val, test = dataset.get_string_interaction_data(0.25, 0.5)

def build_vocab(df_list): 
    entities = set()
    relations = set()
    for df in df_list :
        for _, row in df.iterrows() : 
            entities.add(row["head"])
            entities.add(row["tail"])
            relations.add(row["relation"])
    ent2id = {e : i for i, e in enumerate(sorted(entities))}
    rel2id = {r : i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id
#ent2id, rel2id = build_vocab([train, test, val])

# get mapings and convert data to tensor
def convert_to_tensor(df, ent2id, rel2id,  device) :
    heads = df["head"].map(ent2id).values
    rels = df["relation"].map(rel2id).values
    tails = df["tail"].map(ent2id).values
    
    return torch.LongTensor(heads).to(device), \
        torch.LongTensor(rels).to(device), \
        torch.LongTensor(tails).to(device)

#heads, rels, tails = convert_to_tensor(test, ent2id, rel2id, device)

# corrupt the files
def negative_sample(h, r, t, num_entities) :
    batch_size = h.size(0)
    corrupt_h = torch.randint(0, 2, (batch_size, ), device = h.device)
    randd_ents = torch.randint(0, num_entities, (batch_size, ), device = h.device)

    neg_h = h.clone()
    neg_t = t.clone()
    neg_h[corrupt_h == 0] = randd_ents[corrupt_h == 0]
    neg_t[corrupt_h == 1] = randd_ents[corrupt_h == 1]

    return neg_h, r, neg_t

#nh, nr, nt = negative_sample(heads, rels, tails, len(ent2id))

def train_model(model, train_loader, 
                num_entities, epochs, lr = 0.001) :
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    train_losses = []
    accuracy = []

    def train_test_time (start: float, end: float) -> str:
        run_time = end - start
        return f"{run_time:.2f} seconds"

    model.train()
    for epoch in tqdm(range(epochs)):
        start_time = timer()
        print(f"epoch: {epoch}\n---------------")
        total_loss, total_acc = 0, 0

        for batch, (batch_h, batch_r, batch_t) in enumerate(train_loader) : 
            # get positive scores
            pos_score = model.score(batch_h, batch_r, batch_t)
            
            # negative score
            neg_h, neg_r, neg_t = negative_sample(batch_h, batch_r, batch_t, num_entities)
            neg_score = model.score(neg_h, neg_r, neg_t)

            loss = torch.mean(torch.relu(1.0 - pos_score + neg_score))
            acc = (pos_score > neg_score).float().mean()
            total_acc += acc
            acc = acc * 100

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # if batch % 1000 == 0:
            #     tqdm.write(f"Locked at {batch * len(batch_h)}/{len(train_loader.dataset)} samples")
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        train_losses.append(avg_loss)
        accuracy.append(avg_acc)
        end_time = timer()
        run_time = train_test_time(start_time, end_time)
        tqdm.write(f"epoch {epoch} | loss {loss} | total loss {total_loss} | accuracy = {acc:.2f}% | runtime : {run_time}")

    return train_losses

train_df, val_df, test_df = None, None, None
ent2id, rel2id = None, None
def main() :
    print(f"working on device : {device}")
    global train_df, val_df, test_df, ent2id, rel2id
    train_df, val_df, test_df = dataset.get_string_interaction_data(0.25, 0.15)#.get_triplet_data()#
    ent2id, rel2id = build_vocab([train_df, val_df, test_df])

    n_entities = len(ent2id)
    n_relations = len(rel2id)
    print(f"Entities: {n_entities} | Relations: {n_relations}")

    train_h, train_r, train_t = convert_to_tensor(train_df, ent2id, rel2id, device)
    # val_h, val_r, val_t = convert_to_tensor(val_df, ent2id, rel2id, device)
    # test_h, test_r, test_t = convert_to_tensor(test_df, ent2id, rel2id, device)

    global train_loader
    train_dataset = dataset.KGDataset(train_h, train_r, train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    model = kgeModel.TransE(n_entities, n_relations, 50, device=device)
    print(f"dataloader size, batch size is 64 {len(train_loader.dataset)}")
    train_losses = train_model(model, train_loader, 
            n_entities, epochs=25, lr=0.01)
    print(train_losses)
    return model


if __name__ == "__main__" :
   model = main()

extractor = embedding_extractor.EmbeddingExtractor(model, ent2id, rel2id, device)
visualizer = embedding_visualization.EmbeddingVisualizer(extractor)
print("plot entity embedding")
fig = visualizer.plot_entity_embeddings(method = "tsne")
fig.show()

train_h, train_r, train_t = convert_to_tensor(train_df, ent2id, rel2id, device)
train_triplets = list(zip(
    train_h.cpu().numpy(),
    train_r.cpu().numpy(),
    train_t.cpu().numpy()
))

print("system querry")
query_system = embedding_visualization.KnowledgeGraphQuery(model, extractor, 
                    train_triplets, device)
query = query_system.kg
query["Tnnt1"]["intersectswith"]
similarity = query_system.find_similar_entities("Tnnt1", 5)
list(rel2id.keys())[0]
predict_tail = query_system.predict_tail("Tnnt1", list(rel2id.keys())[0], 5)
#analogy_query = query_system.analogy_query("anguilla", "neighbor", "cuba")

print("classification system")
classifier = embedding_visualization.EmbeddingClassifier(extractor)
list(train_df["head"])[0:20]
# dictt = {"slovakia": "country", "africa" : "continent", 
#  "niger"   : "country", 
#  "paris"   : "city", "cuba": "country",
#  "europe"  : "continent"}
# X, y, names = classifier.prepare_classification_data(dictt)
# clf, train_acc, test_acc = classifier.train_classifier(X, y)

print("system analysis")
analyzer = embedding_visualization.EmbeddingAnalyzer(extractor)
labels, clusters, silhouette = analyzer.cluster_entities(n_clusters=3)
analyzer.compute_embedding_statistics()
analyzer.analyze_embedding_norms()
plt.show()

save_dir = '/kaggle/working/knowledge-graph-for-protein-protein-interactions/'
extractor.save_embeddings(os.path.join(save_dir, 'embeddings.pickle'))