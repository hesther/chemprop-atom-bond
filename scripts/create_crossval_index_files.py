import pickle
import os
from copy import deepcopy
import random
from argparse import ArgumentParser

def create_crossval_indices(args):
    random.seed(0)
    if args.test_folds_to_test is None:
        args.test_folds_to_test = args.num_folds
    if args.val_folds_per_test is None:
        args.val_folds_per_test = args.num_folds - 1
    folds = list(range(args.num_folds))
    random.shuffle(folds)
    os.makedirs(args.save_dir, exist_ok=True)
    for i in folds[:args.test_folds_to_test]:
        with open(os.path.join(args.save_dir, f'{i}.pkl'), 'wb') as wf:
            index_sets = []
            index_folds = deepcopy(folds)
            index_folds.remove(i)
            random.shuffle(index_folds)
            for val_index in index_folds[:args.val_folds_per_test]:
                train, val, test = [index for index in index_folds if index != val_index], [val_index], [val_index] # test set = val set during cv for now
                index_sets.append([train, val, test])
            pickle.dump(index_sets, wf)
        print(i, index_sets)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory to save indices')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number of cross validation folds')
    parser.add_argument('--test_folds_to_test', type=int, 
                        help='Number of cross validation folds to test as test folds')
    parser.add_argument('--val_folds_per_test', type=int, 
                        help='Number of cross validation folds')
    args = parser.parse_args()

    create_crossval_indices(args)