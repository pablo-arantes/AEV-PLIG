# AEV-PLIG

## Conda environment
conda env create --file aev-plig-mac.yml

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
