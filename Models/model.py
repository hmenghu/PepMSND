import os
from modelsfusion import CModel
from Transform import TransformerModel
from KAN import KanMolde
from SE3 import SE3Transformer
from GAT import GATModel
from pretreatment import func
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
import torch.optim
from dgllife.utils import *
from torchmetrics.classification import BinaryAccuracy, Precision, Recall, F1Score
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import warnings

warnings.filterwarnings('ignore')


def main_(batch_size):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    torch.cuda.empty_cache()

    for num in range(1,2):  

        PATH_x_train = f'../Dataset/X_train{num}.csv'
        PATH_x_test = f'../Dataset/X_test{num}.csv'

        (X_transform_train, y_transform_train, X_kan_train, y_kan_train, features_train, y_features_train,
         X_SE3_train, y_SE3_train) = func(PATH_x_train, is_train=True)

        (X_transform_test, y_transform_test, X_kan_test, y_kan_test, features_test, y_features_test,
         X_SE3_test, y_SE3_test) = func(PATH_x_test, is_train=False)

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
                graph = Data(x=x, edge_index=edge_index)  
                graphs.append(graph)

            return graphs, labels

        gat_model = GATModel(input_dim=74, hidden_dim=64, output_dim=128).to(device)
        transform_model = TransformerModel().to(device)
        kan_model = KanMolde().to(device)
        se3_model = SE3Transformer(hidden_dim=128, num_layers=3, num_heads=4, cutoff_distance=5.0).to(device)
        model = CModel().to(device)

        def collate(sample):
            X_transform, X_kan, graphs, labels, features, X_SE3, index = map(list, zip(*sample))

            batched_graph = Batch.from_data_list(graphs)  

            features = torch.stack([torch.tensor(feat, dtype=torch.float32) for feat in features], dim=0)

            X_SE3 = [coords.to(device) for coords in X_SE3]

            labels = torch.tensor(labels)

            return X_transform, X_kan, batched_graph, labels, features, X_SE3, index

        train_X = pd.read_csv(PATH_x_train)
        train_graphs, train_labels = get_data(PATH_x_train)

        train_data = list(zip(X_transform_train, X_kan_train, train_graphs, train_labels, features_train, X_SE3_train,
                              [i for i in range(len(train_X))]))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)

        test_X = pd.read_csv(PATH_x_test)
        test_graphs, test_labels = get_data(PATH_x_test)

        test_data = list(zip(X_transform_test, X_kan_test, test_graphs, test_labels,  features_test, X_SE3_test,
                             [i for i in range(len(test_X))]))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)

        optimizer = torch.optim.AdamW([
            {'params': gat_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': transform_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': kan_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': se3_model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': model.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
        ])  

        accuracy = BinaryAccuracy(threshold=0.5).to(device)
        precision = Precision(task='binary', threshold=0.5).to(device)
        recall = Recall(task='binary', threshold=0.5).to(device)
        f1_score = F1Score(task='binary', threshold=0.5).to(device)

        def save_predictions(predictions, true_values, epoch, phase):
            df = pd.DataFrame({'predict': predictions, 'true': true_values})
            os.makedirs(f'./result_data/{num}/{phase}', exist_ok=True)
            df.to_csv(f'./result_data/{num}/{phase}/{epoch}.csv', index=False)

        for epoch in range(1, 201):
            # train
            gat_model.train()
            transform_model.train()
            kan_model.train()
            se3_model.train()
            model.train()
            train_epoch_loss = 0
            train_predictions = []
            train_true_values = []

            for i, (X_transform, X_kan, graphs, labels, features, X_SE3, index) in enumerate(train_loader):

                train_labels = labels.to(device)

                train_graphs = graphs.to(device)
                pre_y_gat = gat_model(train_graphs.x, train_graphs.edge_index, train_graphs.batch)

                X_transform = torch.cat(X_transform, dim=0)
                num_change = int(len(X_transform) / 1024)
                X_transform = torch.reshape(X_transform, [num_change, 1024])
                X_transform = X_transform.to(device)
                pre_y_transform = transform_model(X_transform)


                X_kan = torch.tensor([item.cpu().detach().numpy() for item in X_kan]).to(device)
                pre_y_kan = kan_model(X_kan)
                
                batch_indices = torch.cat([
                    torch.full((coords.size(0),), i, dtype=torch.long, device=device) for i, coords in enumerate(X_SE3)
                ])
                merged_X_SE3 = torch.cat(X_SE3, dim=0)
                pre_y_se3 = se3_model(merged_X_SE3, batch_indices)

                pre_y = model(pre_y_transform, pre_y_gat, pre_y_kan, pre_y_se3, features).to(device)
                pre_y = torch.reshape(pre_y, [num_change])

                train_loss = nn.BCELoss()(pre_y, train_labels.float())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.detach().item()

                train_predictions.extend(pre_y.detach().cpu().numpy())
                train_true_values.extend(train_labels.detach().cpu().numpy())

            train_acc = accuracy(torch.tensor(train_predictions), torch.tensor(train_true_values))
            train_prec = precision(torch.tensor(train_predictions), torch.tensor(train_true_values))
            train_rec = recall(torch.tensor(train_predictions), torch.tensor(train_true_values))
            train_f1 = f1_score(torch.tensor(train_predictions), torch.tensor(train_true_values))

            save_predictions(train_predictions, train_true_values, epoch, 'train')

            train_epoch_loss /= (i + 1)

            train_binary_preds = [1 if p >= 0.5 else 0 for p in train_predictions]
            train_auc = roc_auc_score(train_true_values, train_predictions)
            train_mcc = matthews_corrcoef(train_true_values, train_binary_preds)

            gat_model.eval()
            transform_model.eval()
            kan_model.eval()
            se3_model.eval()
            model.eval()
            test_epoch_loss = 0
            test_predictions = []
            test_true_values = []
            with torch.no_grad():
                for i, (X_transform, X_kan, graphs, labels, features, X_SE3, index) in enumerate(test_loader):
                    test_labels = labels.to(device)

                    test_graphs = graphs.to(device)
                    pre_y_gat = gat_model(test_graphs.x, test_graphs.edge_index, test_graphs.batch)

                    X_transform = torch.cat(X_transform, dim=0)
                    num_change = int(len(X_transform) / 1024)
                    X_transform = torch.reshape(X_transform, [num_change, 1024])
                    X_transform = X_transform.to(device)
                    pre_y_transform = transform_model(X_transform)

                    X_kan = torch.tensor([item.cpu().detach().numpy() for item in X_kan]).to(device)
                    pre_y_kan = kan_model(X_kan)

                    batch_indices = torch.cat([
                        torch.full((coords.size(0),), i, dtype=torch.long, device=device) for i, coords in enumerate(X_SE3)
                    ])
                    merged_X_SE3 = torch.cat(X_SE3, dim=0)
                    pre_y_se3 = se3_model(merged_X_SE3, batch_indices)

                    pre_test_y = model(pre_y_transform, pre_y_gat, pre_y_kan, pre_y_se3, features).to(device)
                    pre_test_y = torch.reshape(pre_test_y, [num_change])

                    test_loss = nn.BCELoss()(pre_test_y, test_labels.float())
                    test_epoch_loss += test_loss.detach().item()

                    test_predictions.extend(pre_test_y.detach().cpu().numpy())
                    test_true_values.extend(test_labels.detach().cpu().numpy())
                    

                test_acc = accuracy(torch.tensor(test_predictions), torch.tensor(test_true_values))
                test_prec = precision(torch.tensor(test_predictions), torch.tensor(test_true_values))
                test_rec = recall(torch.tensor(test_predictions), torch.tensor(test_true_values))
                test_f1 = f1_score(torch.tensor(test_predictions), torch.tensor(test_true_values))

                save_predictions(test_predictions, test_true_values, epoch, 'test')

                test_epoch_loss /= (i + 1)

                test_binary_preds = [1 if p >= 0.5 else 0 for p in test_predictions]
                test_auc = roc_auc_score(test_true_values, test_predictions)
                test_mcc = matthews_corrcoef(test_true_values, test_binary_preds)

                # save_path = f'./model/{num}/'
                # os.makedirs(save_path, exist_ok=True)

                # torch.save(gat_model, f'{save_path}gat_model_epoch_{epoch}.pt')
                # torch.save(transform_model, f'{save_path}transform_model_epoch_{epoch}.pt')
                # torch.save(kan_model, f'{save_path}kan_model_epoch_{epoch}.pt')
                # torch.save(se3_model, f'{save_path}se3_model_epoch_{epoch}.pt')
                # torch.save(model, f'{save_path}model_epoch_{epoch}.pt')

                print(f"Epoch: {epoch}, Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Train Prec: {train_prec:.4f}, Train Rec: {train_rec:.4f}, Train F1: {train_f1:.4f}, "
                      f"Train AUC: {train_auc:.4f}, Train MCC: {train_mcc:.4f} "
                      f"Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_acc:.4f}, "
                      f"Test Prec: {test_prec:.4f}, Test Rec: {test_rec:.4f}, Test F1: {test_f1:.4f}, "
                      f"Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}")


if __name__ == '__main__':
    batch_size=16
    main_(batch_size)

