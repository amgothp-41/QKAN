# -*- coding: utf-8 -*-
"""
1. Training and validation accuracy 
2.	Training and validation loss
3.	Confusion matrix
4.	Accuracy 
5.	Precision
7.	F-Measure 
9.	Receiver Operating Characteristic 


"""



from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import numpy as np

def compute_mcc(TP, TN, FP, FN):
    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)

    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    # Safe division (handles zero automatically)
    mcc = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)

    return mcc


def Calculation(Y_test, Y_pred):
    
    cnf_matrix = confusion_matrix(Y_test, Y_pred)
    

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)    
    ACC = (TP+TN)/(TP+FP+FN+TN)
    Accuracy = ACC
    recall = (1-FNR)
    precision = (1-FDR)
    err=1-Accuracy
    f1_score = (2*precision*recall)/(precision+recall)
    Specificity = TN / (TN + FP)
    mcc=compute_mcc(TP, TN, FP, FN)

    keys = ['CM',  'Accuracy',  'Precision','recall','Error',
             'F1-Score','Specificity']
    values = [cnf_matrix, Accuracy, precision,recall,err,f1_score,Specificity]
    out = dict(zip(keys, values))
    return out

def main(y_test,ypred):
    import pandas as pd
    import matplotlib.pyplot as plt
    import itertools
    import pickle
    import warnings
    plt.rcParams['font.family'] = 'Times new roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    
    with open("Model/testing_lbls.pkl", "rb") as f:
        df = pickle.load(f)
    
        out0=Calculation(df['ytest'], df['ypred'])
        out1=Calculation(df['ytest'], df['ypred1'])
        out2=Calculation(df['ytest'], df['ypred2'])
        out3=Calculation(df['ytest'], df['ypred3'])
           
        
        # Suppress specific seaborn warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        figpath="Result/"
        
        methods = ['LR-GNB-GS', 'DL-CNN', 'SARSA-OCWLS', 'Proposed']  # Added 5th to match your 'out'
        class_labels = ['BRCA','LUAD','COAD','KIRC','PRAD']
        outs = [ out3, out2, out1, out0]  # list of your outputs
        
        # %% Accuracy
        
        # Initialize dictionary
        values = {}
        
        for i, label in enumerate(class_labels):
            values[label] = {}  # initialize inner dict
            for j, method in enumerate(methods):
                values[label][method] = outs[j]['Accuracy'][i] * 100
        
        df = pd.DataFrame(values)
        # Transpose to get methods as rows and categories as columns
        df = df.T
        # Save to CSV file
        df.to_csv(figpath+"/Accuracy.csv", index=True)
        
        # Plot bar chart
        df.plot(kind="bar", figsize=(7, 5), rot=0, width=0.6)
        
        plt.legend(ncol=2, loc='lower right')
        plt.xticks( rotation=0)
        plt.ylabel('Accuracy (%)')
        plt.xlabel(' ')
        plt.ylim(50, 100)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figpath+"/Accuracy.jpg", dpi=600)
        
        # Show the plot
        plt.show()
        
        
        # %% Precsion
        
        # Define the specific values for each category and legend
        values = {}
        
        for i, label in enumerate(class_labels):
            values[label] = {}  # initialize inner dict
            for j, method in enumerate(methods):
                values[label][method] = outs[j]['Precision'][i] * 100
        df = pd.DataFrame(values)
        # Transpose to get methods as rows and categories as columns
        df = df.T
        # Save to CSV file
        df.to_csv(figpath+"/Precision.csv", index=True)
        
        # Plot bar chart
        df.plot(kind="bar", figsize=(7, 5), rot=0, width=0.6)
        plt.legend(ncol=2, loc='lower right')
        plt.xticks( rotation=0)
        plt.ylabel('Precision (%)')
        plt.xlabel(' ')
        plt.ylim(50, 100)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figpath+"/Precision.jpg", dpi=600)
        
        # Show the plot
        plt.show()
        
        # %% Recall
        
        
        # Define the specific values for each category and legend
        values = {}
        
        for i, label in enumerate(class_labels):
            values[label] = {}  # initialize inner dict
            for j, method in enumerate(methods):
                values[label][method] = outs[j]['recall'][i] * 100
        df = pd.DataFrame(values)
        # df = df['Low Risk'].sort_values(ascending=False)
        # df = df['High Risk'].sort_values(ascending=False)
        # Transpose to get methods as rows and categories as columns
        df = df.T
        # Save to CSV file
        df.to_csv(figpath+"/recall.csv", index=True)
        
        # Plot bar chart
        df.plot(kind="bar", figsize=(7, 5), rot=0, width=0.6)
        plt.legend(ncol=2, loc='lower right')
        plt.xticks( rotation=0)
        plt.ylabel('Recall (%)')
        plt.xlabel(' ')
        plt.ylim(50, 100)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figpath+"/recall.jpg", dpi=600)
        
        # Show the plot
        plt.show()
           
        
        # %% F1-Score
        
        values = {}
        
        for i, label in enumerate(class_labels):
            values[label] = {}  # initialize inner dict
            for j, method in enumerate(methods):
                values[label][method] = outs[j]['F1-Score'][i] * 100
        df = pd.DataFrame(values)
        
        # Transpose to get methods as rows and categories as columns
        df = df.T
        # Save to CSV file
        df.to_csv(figpath+"/F1-Score.csv", index=True)
        
        # Plot bar chart
        df.plot(kind="bar", figsize=(7, 5), rot=0, width=0.6)
        plt.legend(ncol=2, loc='lower right')
        plt.xticks( rotation=0)
        plt.ylabel('F1-Score (%)')
        plt.xlabel(' ')
        plt.ylim(50, 100)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figpath+"/F1-Score.jpg", dpi=600)
        
        # Show the plot
        plt.show()
        # %% Error
        
        values = {}
        
        for i, label in enumerate(class_labels):
            values[label] = {}  # initialize inner dict
            for j, method in enumerate(methods):
                values[label][method] = outs[j]['Error'][i]
        df = pd.DataFrame(values)
        
        # Transpose to get methods as rows and categories as columns
        df = df.T
        # Save to CSV file
        df.to_csv(figpath+"/Error.csv", index=True)
        
        # Plot bar chart
        df.plot(kind="bar", figsize=(7, 5), rot=0, width=0.6)
        plt.legend(ncol=2, loc='lower right')
        plt.xticks( rotation=0)
        plt.ylabel('Prediction Error')
        # plt.xlabel(' ')
        # plt.ylim(50, 100)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(figpath+"/Error.jpg", dpi=600)
        
        # Show the plot
        plt.show()
        # %%
        
        
        def plot_confusion_matrix(cm, classes, savepath, normalize=False, title='', cmap=plt.cm.Greens):
            import numpy as np
          
            fig_size = (7,5)
            plt.figure(figsize=fig_size)
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, fontweight='bold',
                       fontname="Times New Roman", fontsize=16,rotation=90)
            plt.yticks(tick_marks, classes, fontname="Times New Roman",
                       fontsize=16, fontweight='bold')
            fmt = '.2f' if normalize else 'd'
        
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black", fontweight='bold',fontname="Times New Roman", fontsize=16)
            plt.tight_layout()
            plt.savefig(savepath, dpi=600)
            
        
        plot_confusion_matrix(out0['CM'], class_labels,figpath+'\CM_test.JPG')
        
        
            # %%
        
        from sklearn.metrics import auc
        import numpy as np
        # %%
           
        recall = np.array( [np.mean(outs[0]['recall']),np.mean(outs[1]['recall']),np.mean(outs[2]['recall']),np.mean(outs[3]['recall'])])
        specificity = np.array( [np.mean(outs[0]['Specificity']),np.mean(outs[1]['Specificity']),np.mean(outs[2]['Specificity']),np.mean(outs[3]['Specificity'])])
        
        
        # Convert % → 0-1 range
        tpr = recall 
        fpr = 1 - (specificity )
        
        # =========================
        # Plot ROC-like curve
        # =========================
        plt.figure()
        
        for i, method in enumerate(methods):
            # Construct ROC curve points
            roc_fpr = [0, fpr[i], 1]
            roc_tpr = [0, tpr[i], 1]
        
            # AUC calculation
            roc_auc = auc(roc_fpr, roc_tpr)
        
            # Plot
            plt.plot(roc_fpr, roc_tpr,
                     marker='o',
                     linewidth=2,
                     label=f"{method} (AUC = {roc_auc*100:.2f})")
        
        # Random classifier
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        # Formatting
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        
        plt.savefig("Result/roc.jpg", dpi=600, bbox_inches='tight')
    
        
      
