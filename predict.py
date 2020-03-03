"""Loads a trained model checkpoint and makes predictions on a dataset."""
import os
import pickle

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions

import pandas as pd

if __name__ == '__main__':
    args = parse_predict_args()

    test_df = pd.read_csv(args.test_path, index_col=0)
    smiles = test_df.smiles.tolist()
    test_preds, test_smiles = make_predictions(args, smiles=smiles)

    partial_charge = test_preds[0]
    partial_neu = test_preds[1]
    partial_elec = test_preds[2]
    NMR = test_preds[3]

    bond_order = test_preds[4]
    bonnd_indices = test_preds[5]
    df = pd.DataFrame({'smiles':smiles, 'partial_charge':partial_charge, 'partial_neu':partial_neu, 'partial_elec':partial_elec, 'NMR':NMR, 'bond_order':bond_order, 'bond_indices':bonnd_indices})

    with open(os.path.join(args.save_dir, 'preds.pickle'), 'wb') as preds:
        pickle.dump(test_preds, preds)

    with open(os.path.join(args.save_dir, 'preds_smiles.pickle'), 'wb') as smiles:
        pickle.dump(test_smiles, smiles)

    df.to_pickle(os.path.join(args.save_dir, 'predicts.pickle'))

    
