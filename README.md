# AEV-PLIG

AEV-PLIG is a GNN-based scoring function that predicts the binding affinity of a bound protein-ligand complex given its 3D structure.

- [Installation guide](#installation-guide)
- [Demo](#demo)

## Installation guide

### Create conda environment
For *macOS*:
```
conda env create --file aev-plig-mac.yml
```
For *Linux*:
```
conda env create --file aev-plig-linux.yml
```
Install packages manually:
```
conda create --name aev-plig python=3.8
conda activate aev-plig
pip install torch torchvision torchaudio
pip install torch_geometric
pip install rdkit
pip install torchani
pip install qcelemental
pip install pandas
```

## Demo

### Training

#### Download training data
Download the training datasets PDBbind and BindingNet
```
wget http://pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz
wget http://pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database
```
Put PDBbind data into *data/pdbbind/refined-set* and *data/pdbbind/general-set*

Put BindingNet data into *data/bindingnet/from_chembl_client*

#### Generate PDBbind and BindingNet graphs
The following scripts will generate graphs into *pdbbind.pickle* and *bindingnet.pickle*. Takes around 30 minute in total to run.
```
python generate_pdbbind_graphs.py
python generate_bindingnet_graphs.py
```

#### Generate data for pytorch
Running this script takes around 2 minutes.
```
python create_pytorch_data.py
```
The script outputs the following files in *data/processed/*:

*pdbbind_U_bindingnet_ligsim90_train.pt*, *pdbbind_U_bindingnet_ligsim90_valid.pt*, and *pdbbind_U_bindingnet_ligsim90_test.pt*

#### Run training


## Processing
Run data_processing.py

Needs .csv file data/dataset.csv with columns unique_id, pK, sdf_file, pdb_file

Saves data/dataset_processed.csv with same columns with datapoints removed if:
1. .sdf file cannot be read by RDkit
2. Molecule contains rare element
3. Molecule has undefined bond type

Also generates graphs and saves as data/graphs.pickle

## Predictions
Run predictions.py

Saves model predictions of all rows in dataset_processed.csv in output/predictions/predictions.csv

## Enriched training
Download PDBbind and BindingNet:
1. wget http://pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz
2. wget http://pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
3. wget http://bindingnet.huanglab.org.cn/api/api/download/binding_database

Put PDBbind data into data/pdbbind/refined-set and data/pdbbind/general-set

Put BindingNet data into data/bindingnet/from_chembl_client

Run both generate_bindingnet_graphs.py and generate_pdbbind_graphs.py

Create enriched training dataset in data/enriched.csv with columns unique_id, pK

Run create_data_for_enriched_training.py

Run enriched_training.py:

python enriched_training.py --activation_function=leaky_relu --batch_size=128 --dataset=pdbbind_U_bindingnet_ligsim90_enriched --epochs=200 --head=3 --hidden_dim=256 --lr=0.00012291937615434127 --model=GATv2Net
