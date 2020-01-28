"""Trains a model on a dataset."""

from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate, run_training
from chemprop.utils import create_logger

if __name__ == '__main__':
    args = parse_train_args()
    args.batch_size = 2
    args.explicit_Hs = True
    args.data_path = 'test.pickle'
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    run_training(args, logger)
