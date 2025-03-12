import pandas as pd
import numpy as np
from tqdm import tqdm
# import h5py
import scipy.io
import pickle
import shutil, os
import os.path as osp

#----------------------------------------------------
from sklearn.model_selection import train_test_split
#----------------------------------------------------
import torch
from torch.utils.data import TensorDataset
import torch_geometric
from torch_geometric.loader import DataLoader
#----------------------------------------------------
from dataprocessing import sequenceEncoding, peptide_sequence_to_graph, dBLOSUM
from model import DeepCPP

import random


import os
import sys
 
#     return data_list_,list_one_hot,list_label


def valid_peptide(cpp_sequence):
    valid_amino_acids = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    validation_peptide = 4
    sequence = cpp_sequence.upper()
    
    if 5 <= len(sequence) <= 30 and all(residue in valid_amino_acids for residue in sequence):
        validation_peptide = 0
    if (5 > len(sequence) or len(sequence)> 30) and all(residue in valid_amino_acids for residue in sequence) == False:
        validation_peptide = 1    
    elif 5 > len(sequence) or len(sequence)> 30:
        validation_peptide = 2
    elif all(residue in valid_amino_acids for residue in sequence) == False:
        validation_peptide = 3

    return validation_peptide
                  
def predict_peptide(cpp_sequence):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mol_class = ""
    molecule_name = pd.Series(cpp_sequence, name='sequence')
    mol_df = pd.concat([molecule_name], axis=1)
    
    data_test_graph = peptide_sequence_to_graph(mol_df,'test')
    X_data_test_one_hot_encoded = sequenceEncoding(dBLOSUM, mol_df,'test') 
    
    X_test_one_hot_tensor = torch.tensor(X_data_test_one_hot_encoded, dtype=torch.float32)
    test_seq_dataset = TensorDataset(X_test_one_hot_tensor)
    
    test_graph_loader = DataLoader(data_test_graph, batch_size=64)
    test_seq_loader = DataLoader(test_seq_dataset, batch_size=64)
    loaded_model = pickle.load(open('Trained_model/DeepCPP1.pkl', 'rb')) 
    
    # Evaluate the model
    loaded_model.eval()
    with torch.no_grad():
        for (graph_batch, seq_batch) in zip(test_graph_loader, test_seq_loader):
            graph_data = graph_batch.to(device)
            seq_data = seq_batch[0].to(device)

            outputs = loaded_model(graph_data, seq_data).view(-1, 1)

            mol_pred = torch.sigmoid(outputs).cpu().numpy()
            
    if mol_pred >= 0.5:
        mol_class = "CPP"
    else:
        mol_class = "Non-CPP"
    return mol_pred, mol_class


import argparse

# Argument parser to take the FASTA file as input
parser = argparse.ArgumentParser(description="Process a FASTA file for peptide predictions.")
parser.add_argument(
    "fasta_file",
    type=str,
    help="Path to the input FASTA file."
)
args = parser.parse_args()

# Specify the path to the FASTA file from the argument
fasta_file = args.fasta_file




# Initialize a dictionary to store sequences
fasta_sequences = {}

# Read the FASTA file
with open(fasta_file, "r") as file:
    sequence_id = None
    sequence = []
    
    for line in file:
        line = line.strip()  # Remove trailing newline characters
        if line.startswith(">"):  # Header line
            if sequence_id:  # Save the previous sequence
                fasta_sequences[sequence_id] = "".join(sequence)
            sequence_id = line[1:]  # Remove ">" and store the ID
            sequence = []  # Reset sequence
        else:
            sequence.append(line)  # Add sequence lines
    if sequence_id:  # Add the last sequence
        fasta_sequences[sequence_id] = "".join(sequence)

item_n = 0
data_dictionary = {}

score = [[]]   
for seq_id, seq in fasta_sequences.items():
    is_valid = valid_peptide(seq)
    if is_valid == 0:
        key = f"entry_{item_n + 1}"
        prediction_score, mol_class = predict_peptide(seq)
        data_dictionary[key] = {
            "ID": seq_id,
            "sequence": seq,
            "prediction_score": prediction_score,
            "class": mol_class
        }
    else:
        key = f"entry_{item_n + 1}"
        mol_class = "Invalid code - "+ str(is_valid)
        data_dictionary[key] = {
            "ID": seq_id,
            "sequence": seq,
            "prediction_score": None,
            "class": mol_class
        }
    item_n = item_n + 1

for entry in data_dictionary.values():
    # Extracting data
    ID = entry['ID']
    sequence = entry['sequence']
    prediction_score = entry['prediction_score']
    cls = entry['class']

    if prediction_score is None:
        formatted_score = "None"
        print(f"ID: '{ID}', sequence: '{sequence}', score: {0}, class: '{cls}'")
    else:
        formatted_score = prediction_score.item()
        print(f"ID: '{ID}', sequence: '{sequence}', score: {formatted_score:.7f}, class: '{cls}'")
