# ChemProp for atomic and bond property predictions
This repository contains multitask constraint message passing neural networks for atomic/bond property predictions as described in the paper [Regio-Selectivity Prediction with a Machine-Learned Reaction Representation and On-the-Fly Quantum Mechanical Descriptors](https://chemrxiv.org/articles/preprint/Regio-Selectivity_Prediction_with_a_Machine-Learned_Reaction_Representation_and_On-the-Fly_Quantum_Mechanical_Descriptors/12907316). This network is modeled after ChemProp as described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Conda](#option-1-conda)
- [Data](#data)
- [Training](#training)
  * [Train/Validation/Test Splits](#train-validation-test-splits)
- [Predicting](#predicting)
- [Trained model](#Trained model)
  * [API](#API)
  * [torchserve](#torchserve)
- [Results](#results)

## Requirements

While it is possible to run all of the code on a CPU-only machine, GPUs make training significantly faster. To run with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation

### Option 1: Conda

The easiest way to install the `chemprop` dependencies is via conda. Here are the steps:

1. Install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html)
2. `cd /path/to/chemprop`
3. `conda env create -f environment.yml`
4. `source activate chemprop` (or `conda activate chemprop` for newer versions of conda)
5. (Optional) `pip install git+https://github.com/bp-kelley/descriptastorus`

The optional `descriptastorus` package is only necessary if you plan to incorporate computed RDKit features into your model (see [Additional Features](#additional-features)). The addition of these features improves model performance on some datasets but is not necessary for the base model.

Note that on machines with GPUs, you may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/).

## Data

In order to train the model, you must provide training data containing molecules (as SMILES strings) and known atomic/bond target values. Targets must be numpy array for corresponding atomic/bond properties.
Our model can train on any number of atomic/bond properties.

The data file must be a **pickle file with a header row**. For example:
```
                              smiles                                  hirshfeld_charges  ...                                 bond_length_matrix                                  bond_index_matrix
0     CNC(=S)N/N=C/c1c(O)ccc2ccccc12  [-0.026644, -0.075508, 0.096217, -0.287798, -0...  ...  [[0.0, 1.4372890960937539, 2.4525543850909814,...  [[0.0, 0.9595, 0.0158, 0.0162, 0.0103, 0.0008,...
2      O=C(NCCn1cccc1)c1cccc2ccccc12  [-0.292411, 0.170263, -0.085754, 0.002736, 0.0...  ...  [[0.0, 1.2158509801073485, 2.2520730233154076,...  [[0.0, 1.6334, 0.1799, 0.0086, 0.0068, 0.0002,...
3  C=C(C)[C@H]1C[C@@H]2OO[C@H]1C=C2C  [-0.101749, 0.012339, -0.07947, -0.020027, -0....  ...  [[0.0, 1.3223632546838255, 2.468055985361353, ...  [[0.0, 1.9083, 0.0179, 0.016, 0.0236, 0.001, 0...
4                     OCCCc1cc[nH]n1  [-0.268379, 0.027614, -0.050745, -0.045047, 0....  ...  [[0.0, 1.4018301850170725, 2.4667588956616737,...  [[0.0, 0.9446, 0.0311, 0.002, 0.005, 0.0007, 0...
5      CC(=N)NCc1cccc(CNCc2ccncc2)c1  [-0.083162, 0.114954, -0.274544, -0.100369, 0....  ...  [[0.0, 1.5137126697008916, 2.4882198180715465,...  [[0.0, 1.0036, 0.0437, 0.0108, 0.0134, 0.0004,......
```
where atomic properties (*e.g.* hirshfeld_charges) must be a 1D numpy array with the oder same as that of atoms in the SMILES string; and bond properties (e.g. bond_length_matrix) must be a 2D numpy array of shape (number_of_atoms × number_of_atoms)  

## Training

To train a model, run:
```
python train.py --data_path <path> --atom_targets <atom targets> --bond_targets <bond targets>
```
where `<path>` is the path to a CSV file containing a dataset, `<atom targets>` is a list of atomic targets to train, which should be consistent with the column name in the **pickle** file 
and `<bond targets>` is a list of bond targets to train.

For example:
```
CUDA_VISIBLE_DEVICES=1 python train.py --log_freq 200 --hidden_size 600 --batch_size 50 --epochs 100 --depth 6 --atom_targets hirshfeld_charges hirshfeld_fukui_neu hirshfeld_fukui_elec NMR --atom_constraints 0 1 1 --bond_targets bond_index_matrix bond_length_matrix --save_smiles_splits --save_dir QM_137k_fukui_out_scope --loss_weights 1 1 1 0.00001 1 1 --data_path data/QM_137k_fukui_scope.pickle --explicit_Hs
```

Notes:
* The model allows multi-task constraints applied to different atomic properties by specifying `--atom_constraints`
* `--explicit_Hs` can be used to train/predict based on all-atoms (including H) molecular graph
* When the scale of different properties are drastically different, `--loss_weights` flag is suggested, which scale loss function for different targets into similar scales.

### Train/Validation/Test Splits

Our code supports random splitting data into train, validation, and test sets.

**Random:** By default, the data will be split randomly into train, validation, and test sets.

Note: By default, random splits the data into 80% train, 10% validation, and 10% test. This can be changed with `--split_sizes <train_frac> <val_frac> <test_frac>`. For example, the default setting is `--split_sizes 0.8 0.1 0.1`. Both also involve a random component and can be seeded with `--seed <seed>`. The default setting is `--seed 0`.

## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a pickle file containing the predictions will be saved.

For example:
```
python predict.py -test_path predict.csv --checkpoint_path trained_model/QM_137k.pt
```

The predict.csv contains SMILES strings to be predicted.
For example:
```angular2
,smiles,compounds_ID
0,CCC,0
1,CCCCC,1
2,CCCCCC,1
```

## Trained model
We provide the checkpoint file for model trained on 137k molecules curated from PubChem and Pistachio, as described in our paper, in this repo (`trained_model/QM_137k.pt`). 
The trained model can be used either through `predict.py` script discussed in the [Predicting](#Predicting) section, or through the API/torchserve provided in the `torchserve` folder.

###API
The predicting function is wrapped in the `torchserve/handler.py` script. Please refer to the `__name__ == '__main__'` section of `handler.py` for details.

###torchserve
The trained model is also accessible through [torchserve](https://github.com/pytorch/serve). A **mar** file including everything to start a torchserve is provided in `torchserve/model_store/descriptors.mar`, which is generated via:
```angular2
torch-model-archiver --model-name descriptors --version 1.0 --serialized-file QM_137k.pt --handler handler.py --export-path model-store --extra-files model.py,mpn.py,ffn.py,featurization,nn_utils.py
```
To start a torchserve:
```
torchserve --start --ncs --model-store model_store --models descriptors.mar
```