"""
This module defines the PathwayRankingHandler for use in Torchserve.
"""

import os

import torch
from ts.torch_handler.base_handler import BaseHandler
from featurization import mol2graph, get_atom_fdim, get_bond_fdim
from rdkit import Chem

import numpy as np
import pandas as pd


class ReactivityDescriptorHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device('cuda:' + str(properties.get('gpu_id')) if torch.cuda.is_available() else 'cpu')

        model_dir = properties.get('model_dir')
        model_pt_path = os.path.join(model_dir, "QM_137k.pt")

        from models import MoleculeModel

        # Load model and args
        state = torch.load(model_pt_path, lambda storage, loc: storage)
        args, loaded_state_dict = state['args'], state['state_dict']
        atom_fdim = get_atom_fdim()
        bond_fdim = get_bond_fdim() + atom_fdim

        self.model = MoleculeModel(args, atom_fdim, bond_fdim)
        self.model.load_state_dict(loaded_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.initalized = True
        print('Model file {0} loaded successfully.'.format(model_pt_path))

    def preprocess(self, smiles):
        mol_graph = mol2graph(smiles)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2br, bond_types = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb, b2br, bond_types = \
            f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), \
            b2revb.to(self.device), b2br.to(self.device), bond_types.to(self.device)

        return f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2br, bond_types

    def inference(self, data):
        descs = self.model(data)

        return descs

    def postprocess(self, inference_output):

        smiles = inference_output['smiles']
        descs = inference_output['descs']

        descs = [x.data.cpu().numpy() for x in descs]

        partial_charge, partial_neu, partial_elec, NMR, bond_order, bond_distance = descs

        n_atoms, n_bonds = [], []
        for s in smiles:
            m = Chem.MolFromSmiles(s)

            m = Chem.AddHs(m)

            n_atoms.append(len(m.GetAtoms()))
            n_bonds.append(len(m.GetBonds()))

        partial_charge = np.split(partial_charge.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        partial_neu = np.split(partial_neu.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        partial_elec = np.split(partial_elec.flatten(), np.cumsum(np.array(n_atoms)))[:-1]
        NMR = np.split(NMR.flatten(), np.cumsum(np.array(n_atoms)))[:-1]

        bond_order = np.split(bond_order.flatten(), np.cumsum(np.array(n_bonds)))[:-1]
        bond_distance = np.split(bond_distance.flatten(), np.cumsum(np.array(n_bonds)))[:-1]

        df = pd.DataFrame(
            {'smiles': smiles, 'partial_charge': partial_charge, 'partial_neu': partial_neu,
             'partial_elec': partial_elec,
             'NMR': NMR, 'bond_order': bond_order, 'bond_distance': bond_distance})

        return df

_service = ReactivityDescriptorHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    if isinstance(data, list):
        smiles = data
    else:
        smiles = data[0].get('data') or data[0].get('body')

    outputs = _service.inference(_service.preprocess(smiles))
    postprocess_inputs = {'smiles': smiles, 'descs': outputs}

    return _service.postprocess(postprocess_inputs)


if __name__ == '__main__':

    class Context(object):
        def __init__(self):
            self.system_properties = {
                'model_dir': '.'
            }

    context = Context()
    data = ['CCCC', 'CCC', 'CCCCC']
    out = handle(data, context)
    print(out)