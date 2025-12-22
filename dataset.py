import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import mygene

class KGDataset(Dataset):
    def __init__(self, heads, rels, tails):
        assert len(heads) == len(rels) == len(tails)
        self.heads = heads
        self.rels = rels
        self.tails = tails

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        return self.heads[idx], self.rels[idx], self.tails[idx]

# create dataframe
def get_triplet_data():
    """Countries neighborhood and locations downloaded from github"""
    train = pd.read_csv("data/train.txt", sep = "\t")
    test = pd.read_csv("data/test.txt", sep = "\t")
    valid = pd.read_csv("data/valid.txt", sep = "\t")

    train.columns = ["head", "relation", "tail"]
    test.columns = ["head", "relation", "tail"]
    valid.columns = ["head", "relation", "tail"]
    return train, test, valid

def get_string_interaction_data(test_size = 0.25, valid_size = 0.15):
    """Here, clean and struncture the ppi dataset from string for KGE, k split param"""

    #path = "data/10090.protein.links.full.v12.0.txt"
    path_kaggle = "/kaggle/input/string-ppi-data-mus-musculus/10090.protein.links.full.v12.0.txt"
    data = pd.read_csv(path_kaggle, sep = " ")
    data.columns
    col_name = ["protein1", "protein2", "database"]
    data = data[col_name]
    data["edge"] = data["database"].apply(lambda x: "intersectswith" if x > 0 else None)
    data = data[data["database"] > 0]
    data = data.drop(columns = "database")
    data.head()


    proteins = set(data["protein1"].tolist() + data["protein2"].tolist())
    proteins_clean = [pid.split(".")[1] for pid in proteins]
    mg = mygene.MyGeneInfo()
    mapped = mg.querymany(proteins_clean, 
                        scopes = "ensembl.protein",
                        fields = "symbol",
                        species=10090,
                        as_dataframe = False)

    protein_to_symbol = {item["query"]: item.get("symbol", item["query"]) for item in mapped}
    data["gene1"] = data["protein1"].apply(lambda x : protein_to_symbol.get(x.split(".")[1], x))
    data["gene2"] = data["protein2"].apply(lambda x : protein_to_symbol.get(x.split(".")[1], x))
    data = data.drop(columns = ["protein1", "protein2"])
    data.rename(columns = {"gene1" : "head", "gene2": "tail", "edge": "relation"}, inplace = True)
    data = data[["head", "relation", "tail"]]
    data = data.reset_index(drop = True)
    train, temp = train_test_split(data, test_size = test_size, shuffle = True, random_state = 42)
    test, valid = train_test_split(temp, test_size = valid_size, random_state = 42)
    return train, test, valid