import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv, SumPooling
import warnings
from torch.utils.data import DataLoader
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_complete_graph
import numpy as np
import pandas as pd
from rdkit import Chem
from torchmetrics.classification import BinaryAccuracy, Precision, Recall, F1Score
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import dgl
import torch.optim as optim

warnings.filterwarnings('ignore')

class GINNet(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GINNet, self).__init__()
        self.conv1 = GINConv(apply_func=nn.Linear(in_feats, hidden_feats), aggregator_type='sum')
        self.conv2 = GINConv(apply_func=nn.Linear(hidden_feats, hidden_feats), aggregator_type='sum')
        self.fc1 = nn.Linear(hidden_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, 1)
        self.pool = SumPooling()
        self.dropout = nn.Dropout(0.5)

    def forward(self, g):
        h = g.ndata['x']
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        hg = self.pool(g, h)
        hg = F.relu(self.fc1(hg))
        hg = self.dropout(hg)
        out = torch.sigmoid(self.fc2(hg))
        return out
def main_():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    for num in range(1, 2):
        PATH_x_train = f'../Dataset/X_train{num}.csv'
        PATH_x_test = f'../Dataset/X_test{num}.csv'

        node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

        def get_data(PATH_x):
            df = pd.read_csv(PATH_x)
            mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
            labels = df['label'].values.astype(np.float32)
            graphs = []
            for mol, label in zip(mols, labels):
                edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
                g = mol_to_complete_graph(mol, node_featurizer=node_featurizer)
                x = g.ndata['h'].float()
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
                graph.ndata['x'] = x
                graphs.append(graph)
            return graphs, labels

        graphs_train, labels_train = get_data(PATH_x_train)
        graphs_test, labels_test = get_data(PATH_x_test)

        def collate(samples):
            graphs, y_true_GIN = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.tensor(y_true_GIN, dtype=torch.float32)

        train_data = list(zip(graphs_train, labels_train))
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate, drop_last=False)

        test_data = list(zip(graphs_test, labels_test))
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate, drop_last=False)

        in_feats = len(train_loader.dataset[0][0].ndata['x'][0])
        hidden_feats = 128

        model = GINNet(in_feats, hidden_feats).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        accuracy = BinaryAccuracy(threshold=0.5).to(device)
        precision = Precision(task='binary', threshold=0.5).to(device)
        recall = Recall(task='binary', threshold=0.5).to(device)
        f1_score = F1Score(task='binary', threshold=0.5).to(device)

        for epoch in range(1, 201):
            model.train()
            train_epoch_loss = 0
            train_predictions = []
            train_true_values = []

            for batched_graph, y_train in train_loader:
                y_train = y_train.to(device)
                
                batched_graph = batched_graph.to(device)
                pre_y_gin = model(batched_graph).squeeze()
                
                train_loss = nn.BCELoss()(pre_y_gin, y_train)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                
                train_predictions.extend(pre_y_gin.cpu().detach().numpy())
                train_true_values.extend(y_train.cpu().detach().numpy())

            train_epoch_loss /= len(train_loader)
            train_binary_preds = [1 if p >= 0.5 else 0 for p in train_predictions]

            train_acc = accuracy(torch.tensor(train_predictions), torch.tensor(train_true_values)).item()
            train_prec = precision(torch.tensor(train_predictions), torch.tensor(train_true_values)).item()
            train_rec = recall(torch.tensor(train_predictions), torch.tensor(train_true_values)).item()
            train_f1 = f1_score(torch.tensor(train_predictions), torch.tensor(train_true_values)).item()
            train_auc = roc_auc_score(train_true_values, train_predictions)
            train_mcc = matthews_corrcoef(train_true_values, train_binary_preds)

            model.eval()
            test_epoch_loss = 0
            test_predictions = []
            test_true_values = []

            with torch.no_grad():
                for batched_graph, y_test in test_loader:
                    y_test = y_test.to(device)
                    
                    batched_graph = batched_graph.to(device)
                    pre_y_gin = model(batched_graph).squeeze()
                    
                    test_loss = nn.BCELoss()(pre_y_gin.squeeze(), y_test)
                    test_epoch_loss += test_loss.item()
                    
                    test_predictions.extend(pre_y_gin.cpu().detach().numpy())
                    test_true_values.extend(y_test.cpu().detach().numpy())

            test_epoch_loss /= len(test_loader)
            test_binary_preds = [1 if p >= 0.5 else 0 for p in test_predictions]

            test_acc = accuracy(torch.tensor(test_predictions), torch.tensor(test_true_values)).item()
            test_prec = precision(torch.tensor(test_predictions), torch.tensor(test_true_values)).item()
            test_rec = recall(torch.tensor(test_predictions), torch.tensor(test_true_values)).item()
            test_f1 = f1_score(torch.tensor(test_predictions), torch.tensor(test_true_values)).item()
            test_auc = roc_auc_score(test_true_values, test_predictions)
            test_mcc = matthews_corrcoef(test_true_values, test_binary_preds)

            print(f"Epoch: {epoch}, Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Train Prec: {train_prec:.4f}, Train Rec: {train_rec:.4f}, Train F1: {train_f1:.4f}, "
                  f"Train AUC: {train_auc:.4f}, Train MCC: {train_mcc:.4f}, "
                  f"Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"Test Prec: {test_prec:.4f}, Test Rec: {test_rec:.4f}, Test F1: {test_f1:.4f}, "
                  f"Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}")

if __name__ == '__main__':
    main_()