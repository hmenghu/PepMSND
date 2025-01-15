import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score

def main_():

    for num in range(1, 2): 
        PATH_x_train = f'../Dataset/X_train{num}.csv'
        PATH_x_test = f'../Dataset/X_test{num}.csv'  
        
        def create_dataset(PATH_x):
            df = pd.read_csv(PATH_x)
            X = df.drop(columns=['ID', 'PMID', 'SMILES', "label", "Length", "SE-3", 'Species', 'Environment']).values
            scale_parameter = pd.read_csv("../scale_parameter/scale_parameter1.csv")
            mean = scale_parameter['Mean'].values
            std = scale_parameter['Std'].values     
            X = (X - mean) / std 
            y = df['label'].values
            
            return X, y
        
        X_train, y_train = create_dataset(PATH_x_train)
        X_test, y_test = create_dataset(PATH_x_test)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'eta': 0.3,
                'seed': 3407
                }

        model = xgb.train(params, dtrain, num_boost_round=100)


        y_pred_proba = model.predict(dtest) 
        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main_()
