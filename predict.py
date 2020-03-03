"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions

if __name__ == '__main__':
    args = parse_predict_args()
    test_preds, test_smiles = make_predictions(args)

    with open(os.path.join(args.save_dir, 'preds.pickle'), 'wb') as preds:
        pickle.dump(test_preds, preds)

    with open(os.path.join(args.save_dir, 'preds_smiles.pickle'), 'wb') as smiles:
        pickle.dump(test_smiles, smiles)
