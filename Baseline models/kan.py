from efficient_kan import KANLinear
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as  optim
from sklearn.preprocessing import scale
from torchmetrics.classification import BinaryAccuracy, Precision, Recall, F1Score
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import warnings

warnings.filterwarnings('ignore')

class KanModel(nn.Module):
    def __init__(self):
        super(KanModel, self).__init__()
        self.fc1 = KANLinear(142, 128)
        self.fc2 = KANLinear(128, 64)
        self.fc3 = KANLinear(64,1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(128)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.norm(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sig(out)

        return out
    
def create_dataset_KAN(PATH_x):
    df = pd.read_csv(PATH_x)
    df_kan = df.drop(columns=['ID', 'PMID', 'SMILES', "label", "Length", "SE-3", 'Species', 'Environment']).values
    scale_parameter = pd.read_csv("../scale_parameter/scale_parameter1.csv") 
    mean = scale_parameter['Mean'].values
    std = scale_parameter['Std'].values
    df_kan = (df_kan - mean) / std 
    df_species_env = df[['Species', 'Environment']].values
    df_kan = np.hstack((df_kan, df_species_env))
    y = df['label'].values.reshape(-1, 1).astype('float32')
    df_kan = torch.tensor(df_kan, dtype=torch.float32)
    
    return df_kan, y

def main_():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    for num in range(1, 2):
        PATH_x_train = f'../Dataset/X_train{num}.csv'
        PATH_x_test = f'../Dataset/X_test{num}.csv'
        
        df_kan_train, y_train = create_dataset_KAN(PATH_x_train)
        df_kan_test, y_test =  create_dataset_KAN(PATH_x_test)
        
        train_data = list(zip(df_kan_train, y_train))
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True,  drop_last=False)

        test_data = list(zip(df_kan_test, y_test))
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, drop_last=False)
        
        
        model = KanModel().to(device)
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

            for X_train, y_train in train_loader:
                
                y_train = y_train.to(device)
                
                X_train = X_train.to(device)
                pre_y_kan= model(X_train)
                
                train_loss = nn.BCELoss()(pre_y_kan, y_train)
                
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_predictions.extend(pre_y_kan.cpu().detach().numpy())
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
                for X_test, y_test in test_loader:
                    
                    y_test = y_test.to(device)
                    
                    X_test = X_test.to(device)
                    pre_y_kan = model(X_test)
                    
                    test_loss = nn.BCELoss()(pre_y_kan, y_test)
                    test_epoch_loss += test_loss.item()
                    test_predictions.extend(pre_y_kan.cpu().detach().numpy())
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
