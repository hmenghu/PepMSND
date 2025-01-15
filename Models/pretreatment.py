import torch
import warnings
import numpy as np
import pandas as pd
import re
from MDAnalysis import Universe

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_dataset_KAN(PATH_x, is_train=True):  # Create KAN input 
    df = pd.read_csv(PATH_x)
    df_kan = df.drop(columns=['ID', 'PMID', 'SMILES', "label", "Length", "SE-3", 'Species', 'Environment']).values
    
    if is_train:
        mean = np.mean(df_kan, axis=0)
        std = np.std(df_kan, axis=0, ddof=0)
        
        scale_parameter = pd.DataFrame({'Mean': mean, 'Std': std})
        scale_parameter.to_csv("../scale_parameter/scale_parameter1.csv", index=False)
        
        df_kan = (df_kan - mean) / std

    else:
        scale_parameter = pd.read_csv("../scale_parameter/scale_parameter1.csv")
        mean = scale_parameter['Mean'].values
        std = scale_parameter['Std'].values
        
        df_kan = (df_kan - mean) / std 
    
    df_species_env = df[['Species', 'Environment']].values
    df_kan = np.hstack((df_kan, df_species_env))
    y = df['label'].values.reshape(-1, 1).astype('float32')
    
    return df_kan, y

def get_feature(PATH_x):  # Create species and environment features input
    df = pd.read_csv(PATH_x)
    columns = ['Species', 'Environment']
    features = df[columns].values
    features = torch.tensor(features, dtype=torch.float32)
    y = df['label'].values.reshape(-1, 1).astype('float32')
    return features, y


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = [line.strip() for line in f.readlines()]
    return vocab

def create_dataset_transform(PATH_x, vocab_path="../Vocab.txt"):  
    
    vocab = load_vocab(vocab_path)
    word2id = {d: i for i, d in enumerate(vocab)}

    df = pd.read_csv(PATH_x, usecols=['SMILES'])
    datas = []

    for _, row in df.iterrows():
        data = row["SMILES"]
        tokens = smi_tokenizer(data).split(" ")
        if len(tokens) <= 1024:
            di = tokens + ["PAD"] * (1024 - len(tokens))
        else:
            di = tokens[:1024]
        datas.append(di)

    mlist = []
    for d_i in datas:
        mi = [word2id.get(d, word2id["PAD"]) for d in d_i] 
        mlist.append(np.array(mi))

    return mlist

def extract_coordinates(pdb_file):   # Extract atomic coordinates as input for se-3Transform
    u = Universe(pdb_file)
    coordinates = u.atoms.positions
    return torch.tensor(coordinates, dtype=torch.float32)

def create_dataset_SE3(PATH_x):  # Create SE-3 transform input
    df = pd.read_csv(PATH_x)
    pdb_files = df["SE-3"]
    X_SE3 = []
    y = df['label'].values.reshape(-1, 1).astype('float32')
    for pdb_file in pdb_files:
        coordinates = extract_coordinates(pdb_file)
        X_SE3.append(coordinates)

    return X_SE3, y


def func(PATH, is_train=True):
    
    df_kan, y_kan = create_dataset_KAN(PATH, is_train)

    features, y_features = get_feature(PATH)

    df_transform = create_dataset_transform(PATH)
    df_transform = torch.tensor([item for item in df_transform]).to(torch.int64)

    df_kan = torch.tensor(df_kan, dtype=torch.float32)

    y_transform = torch.tensor([item for item in y_kan]).to(torch.float)

    X_SE3, y_SE3 = create_dataset_SE3(PATH)

    return df_transform, y_transform, df_kan, y_kan, features, y_features, X_SE3, y_SE3

