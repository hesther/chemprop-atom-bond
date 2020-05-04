"""Loads a trained model checkpoint and makes predictions on a dataset."""
import os
import pickle

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions

import pandas as pd
from rdkit import Chem
import numpy as np
import time

def num_atoms_bonds(smiles):
    m = Chem.MolFromSmiles(smiles)

    m = Chem.AddHs(m)

    return len(m.GetAtoms()), len(m.GetBonds())

if __name__ == '__main__':
    args = parse_predict_args()

    test_df = pd.read_csv(args.test_path, index_col=0)
    smiles = test_df.smiles.tolist()
    
    start = time.time()
    test_preds, test_smiles = make_predictions(args, smiles=smiles)
    end = time.time()

    print('time:{}s'.format(end-start))

    partial_charge = test_preds[0]
    partial_neu = test_preds[1]
    partial_elec = test_preds[2]
    NMR = test_preds[3]

    bond_order = test_preds[4]
    bond_distance = test_preds[5]

    n_atoms, n_bonds = zip(*[num_atoms_bonds(x) for x in smiles])

    partial_charge = np.split(partial_charge.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
    partial_neu = np.split(partial_neu.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
    partial_elec = np.split(partial_elec.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
    NMR = np.split(NMR.flatten(), np.cumsum(np.array(n_atoms)))[:-1]

    bond_order = np.split(bond_order.flatten(), np.cumsum(np.array(n_bonds)))[:-1]
    bond_distance = np.split(bond_distance.flatten(), np.cumsum(np.array(n_bonds)))[:-1]

    df = pd.DataFrame(
        {'smiles': smiles, 'partial_charge': partial_charge, 'partial_neu': partial_neu, 'partial_elec': partial_elec,
         'NMR': NMR, 'bond_order': bond_order, 'bond_distance': bond_distance})
    
    df.to_pickle(os.path.join(args.save_dir, 'substitution_all_reactants.pickle'))
