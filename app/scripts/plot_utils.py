import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc_curve(y_true, y_proba, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def plot_pr_curve(y_true, y_proba, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(output_path)
    plt.close()

def plot_prediction_distribution(y_true, y_proba, output_path):
    # Simplified HTML distribution summary
    df = pd.DataFrame({'true': y_true, 'proba': y_proba})
    with open(output_path, 'w') as f:
        f.write("<html><body><h1>Prediction Distribution Summary</h1>")
        f.write(f"<p>Mean Probability: {df['proba'].mean():.4f}</p>")
        f.write(df.groupby('true')['proba'].describe().to_html())
        f.write("</body></html>")
