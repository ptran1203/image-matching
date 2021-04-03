import numpy as np
import pandas as pd


def row_wise_f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        pred = pred[:50]
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores, np.mean(scores)


df = pd.read_csv('valid.csv')
_, score = row_wise_f1_score(df.target, df.preds)

print(score)
