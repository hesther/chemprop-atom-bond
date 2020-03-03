"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions

import pandas as pd

if __name__ == '__main__':
    args = parse_predict_args()

    test_df = pd.read_csv(args.preds_path, index_col=0)
    smiles = test_df.smiles.values
    test_preds, test_smiles = make_predictions(args, smiles=smiles)

    with open(os.path.join(args.save_dir, 'preds.pickle'), 'wb') as preds:
        pickle.dump(test_preds, preds)

    with open(os.path.join(args.save_dir, 'preds_smiles.pickle'), 'wb') as smiles:
        pickle.dump(test_smiles, smiles)
