from sklearn.metrics import (f1_score, roc_auc_score,confusion_matrix, accuracy_score,
                             precision_score, recall_score)

# For reporting the results
from IPython.display import HTML, display
import tabulate

def predict_and_evaluate(model, X_test, y_test):
    '''Predict values for given model & test dataset
    and evaluate the results.'''
    
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    
    metrics = [fp, fn, round(precision,2), round(recall,2), round(f1,2)]
    table_row = [[model.__class__.__name__] + metrics]
    display(HTML(tabulate.tabulate(table_row,headers=('Algorithm','False Positives', 
                                                  'False Negatives', 'Precision', 
                                                  'Recall', 'F1 Score'), 
                                   tablefmt='html')))
    return [model.__class__.__name__] + metrics