"""Trains a model on a dataset."""

from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate, run_training
from chemprop.utils import create_logger
import pickle

if __name__ == '__main__':
    args = parse_train_args()
    args.batch_size = 2
    args.explicit_Hs = True
    args.target_constraints = 0
    args.data_path = 'test.pickle'
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    test_avg_score, test_preds, test_smiles = run_training(args, logger)
    with open('test_preds.pickle', 'wb') as preds:
        pickle.dump(test_preds, preds)

    with open('test_smiles.pickle', 'wb') as smiles:
        pickle.dump(test_smiles, smiles)