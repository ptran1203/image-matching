import numpy as np
import pandas as pd

def _convert_to_numpy(list_of_str):
    return np.array(list_of_str.split(' '))

def row_wise_f1_score(labels, preds):
    scores = []

    if isinstance(labels, str):
        labels = _convert_to_numpy(labels)

    if isinstance(preds, str):
        preds = _convert_to_numpy(preds)
    
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores, np.mean(scores)


df = pd.read_csv('valid.csv')
_, score = row_wise_f1_score(df.target, df.preds)

print(score)
