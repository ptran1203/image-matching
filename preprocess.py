import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='group')
    args, _ = parser.parse_known_args()
    return args

def to_str(row):
    return ' '.join(row)

def main():
    args = parse_args()

    print(args)

    df = pd.read_csv('/content/train_.csv')
    df['label_group'] =  LabelEncoder().fit_transform(df.label_group)
    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)
    df['target'] = df['target'].apply(to_str)

    if args.method == 'group':
        skf = GroupKFold(5, random_state=233)
        df['fold'] = -1
        for i, (train_idx, valid_idx) in enumerate(skf.split(df, None, df['label_group'])):
            df.loc[valid_idx, 'fold'] = i
    else:
        skf = StratifiedKFold(5, shuffle=True, random_state=233)
        for i, (train_idx, valid_idx) in enumerate(skf.split(df, df['label_group'])):
            df.loc[valid_idx, 'fold'] = i

    print(f'Split data {skf.__class__.__name__}')

    for fl in range(5):
        train = df[df.fold != fl]
        val = df[df.fold == fl]
        overlap = len(np.intersect1d(train.label_group.unique(), val.label_group.unique()))
        print(f'Fold {fl}, train {train.label_group.nunique()} classes, val {val.label_group.nunique()} classes, overlap {overlap}')

    df.to_csv('/content/train.csv', index=False)

if __name__ == '__main__':
    main()
