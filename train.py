"""Trains a model on a dataset."""

from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate, run_training
from chemprop.utils import create_logger
import pickle

import os

if __name__ == '__main__':
    args = parse_train_args()
    args.explicit_Hs = True
    args.data_path = 'chembl_results.pickle.gz'
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    test_avg_score, test_preds, test_smiles = run_training(args, logger)
    with open(os.path.join(args.save_dir, 'test_preds.pickle'), 'wb') as preds:
        pickle.dump(test_preds, preds)

    with open(os.path.join(args.save_dir, 'test_smiles.pickle'), 'wb') as smiles:
        pickle.dump(test_smiles, smiles)
