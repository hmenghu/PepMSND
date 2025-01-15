from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, roc_auc_score

def main_():
    for num in range(1, 11): 
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

        clf = SVC(kernel='rbf', probability=True, random_state=3407)
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  
        
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Results for dataset {num}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {pre:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")

if __name__ == "__main__":
    main_()