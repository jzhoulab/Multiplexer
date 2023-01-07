import pyfaidx 
import torch
from torch import nn
import pandas as pd
import importlib

import sys
sys.path.append('../models')
from BelugaMultiplexer import BelugaMultiplexer
from Beluga import Beluga

def get_col_labels(custom = None):
    """
    Returns a list containing file names, used when user makes predictions with custom model and uses --add_tsv
    
    Args:
        custom : str
            Path to file containing custom column names
            
    Returns:
        col_mames : list
            list containing column names
    
    """
    
    if custom:
        with open(custom) as file:
            col_names = [i.strip("\n") for i in file.readlines()]
    else:    
        
        x = pd.read_csv("../CLIdata/2002_features.csv",index_col = 0)
        col_names = []

        for i in range(1,2003):

            if str(x["Treatment"][i]) == 'nan':
                treatment = ""
            else:
                treatment = "|" + x["Treatment"][i]


            column_name = x["Cell type"][i] + "|" + x["Assay"][i]+ treatment

            col_names.append(column_name)
        
    return col_names


def get_row_labels(chrome_num, position, ref_chromosome, seqlen):
    """
    Returns a list of row labels for a given prediction, used when --add_tsv is True
    
    Args:
        chrome_num : str
            The chromosome to be sample from (e.g chr11)
            
        position : int
            Center position of the chromosome to be sample from
            
        ref_chromosome : list
            List containing indices representing the base-pair at each position of the reference sequence
        
        seqlen : int
            Length of user given sequence

    """
    
    
    position = position
    arr = []
    
    for i in range(seqlen):
        position = position + i
        if ref_chromosome[i] == 0:
            arr.append(chrome_num + ":" + str(position) + " A>G")
            arr.append(chrome_num + ":" + str(position) + " A>C")
            arr.append(chrome_num + ":" + str(position) + " A>T")

        elif ref_chromosome[i] == 1:
            arr.append(chrome_num + ":" + str(position) + " G>A")
            arr.append(chrome_num + ":" + str(position) + " G>C")
            arr.append(chrome_num + ":" + str(position) + " G>T") 

        elif ref_chromosome[i] == 2:
            arr.append(chrome_num + ":" + str(position) + " C>A")
            arr.append(chrome_num + ":" + str(position) + " C>G")
            arr.append(chrome_num + ":" + str(position) + " C>T")

        elif ref_chromosome[i] == 3:
            arr.append(chrome_num + ":" + str(position) + " T>A")
            arr.append(chrome_num + ":" + str(position) + " T>C")
            arr.append(chrome_num + ":" + str(position) + " T>G")
        
    return arr




def _reshape_prediction(ref, prediction, predlen, seqlen):

    
    ref = ref[0]
    prediction = prediction[0]
    
    alt_prediction = torch.zeros((predlen, 3, seqlen))
    ignore_indices = []
    
    for i in range(seqlen):
        if ref[:,i].sum() == 0: #in the case the reference nucleotide was 'n'
            ignore_index = 0
        else:
            ignore_index = torch.nonzero(ref[:,i]).item()
        
        index = 0
        ignore_indices.append(ignore_index)
        for j in range(4):
            if j == ignore_index:
                continue
            alt_prediction[:, index, i] = prediction[:,j,i]
            index += 1
            

    return alt_prediction.reshape(predlen, -1).T, ignore_indices


def log_unfold(ref, lg_diff):
    """
    Given the reference predictions and log-fold difference between the reference and alternative predictions, return the alternative value
    
    Args:
        ref : torch.tensor
            Predictions made by the Base model for the reference sequence
            
        lg_diff : torch.tensor
            Log-fold change predictions made by the Multiplexer model
            
    Returns:
        Torch.tensor representing the predicted values of the alternative sequence
    
    """
    
    #given the reference value and lg_difference, returns the alternative value
 
    e = 10**(-6)
    z = torch.exp(lg_diff + e) * (ref + e)/(1 - ref + e)

    return z/(1 + z)


def encode_sequence(chrome_num, position, genome, length):
    """
    Returns a sequence from chromosome chrome_num centered at position pos encoded
    as a 4xlength tensor
    
    Args:
        chrome_num : str
            The chromosome to be sample from (e.g chr11)
            
        position : int
            Center position of the chromosome to be sample from
            
        genome : pyfasta object
            The genome that sequences are drawn from
            
        length : int
            Length of input sequence
            
   Returns:
       seq_encoded : torch.tensor
           encoded version of the input sequence
        

    """
    if length % 2 == 0:
        lower = int(length/2) - 1
        upper = int(length/2)
        
    else:
        lower = int(length/2)
        upper = int(length/2)
    
    seq = genome[chrome_num][position - lower -1 : position + upper]

    #define the encoding
    encoding_dict = {'A': torch.tensor([1, 0, 0, 0]), 'G': torch.tensor([0, 1, 0, 0]),
            'C': torch.tensor([0, 0, 1, 0]), 'T': torch.tensor([0, 0, 0, 1]),
            'N': torch.tensor([0, 0, 0, 0]), 'H': torch.tensor([0, 0, 0, 0]),
            'a': torch.tensor([1, 0, 0, 0]), 'g': torch.tensor([0, 1, 0, 0]),
            'c': torch.tensor([0, 0, 1, 0]), 't': torch.tensor([0, 0, 0, 1]),
            'n': torch.tensor([0, 0, 0, 0]), '-': torch.tensor([0, 0, 0, 0])}
    

    
    #create a encoded sequence 
    seq_encoded = torch.zeros((4, len(seq)))
    
    
    for i in range(len(seq)):
        seq_encoded[:,i] = encoding_dict[seq[i]]

        
    return seq_encoded.unsqueeze(0)


    
def predict(inputs_file = None, chrome_num = None, position = None, output_name  = None, add_tsv = None, difference = None, predict_model = None, predict_model_path = None, predict_weights = None, seq_len = None, prediction_dim = None, base_model = None, base_model_path = None, base_weights = None, col_names = None, genome = None, ):
    
    """
    Takes a user-given input and generates an prediction from a Multiplexer model
    
    Args:
        inputs_file : str
            Name of a file that contains multiple chromosome position pairs
            
        chrome_num : str
            The chromosome to be sample from (e.g chr11)
            
        position : int
            Center position of the chromosome to be sample from
            
        output_name : str
            Name of file containing saved predictions
            
        add_tsv : bool
            True if user wants to save predictions in tsv format
            
        difference : bool
            True if user wants to predict the difference between alternative and reference allele instead of the log-fold change
            
        predict_model : str
            Name of the user's mutliplexer model
            
        predict_model_path : str
            Path to file containing user's multiplexer model
            
        predict_weights : str
            Path to file containing user's multiplexer model weights
            
        seq_len : int
            Length of user given sequence
            
        prediction_dim : int
            Number of predicted features made by the user model (e.g For BelugaMultiplexer 2002 features are predicted)
            
        genome : str
            Path of user's custom genome
            
        base_model : str
            Name of user's base model (necessary if difference==True and a custom model is input)
            
        base_model_path : str
            Path to file containing user's Base model (necessary if difference==True and a custom model is input)
            
        base_weights : str
            Path to file containing user's Base model weights
            
        col_names : str
            Path to a file containing a user's custom column names (used if add_tsv==True)

    
    """
    if predict_model:
        spec = importlib.util.spec_from_file_location(predict_model, predict_model_path)
        model_class = getattr(spec.loader.load_module(), predict_model) ##use this
        model = model_class()
        model_weights = predict_weights
       
        
    else:
        model = BelugaMultiplexer() 
        model_weights = '../CLIdata/BelugaMultiplexerWeights.pth'
        
    
    
    #load model and genome
    genome = pyfaidx.Fasta('../CLIdata/hg19.fa') if genome == None else genome
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 2000 if seq_len == None else int(seq_len)
    prediction_dim = 2002 if prediction_dim == None else int(prediction_dim)
   
    
    #positional encodings must equal the i
    enc_length = int(seq_len)
    print("Job completed on: ", device)
    
    #load model weights
    model.load_state_dict(torch.load( model_weights ))
    model.to(device)
    
    
    if inputs_file: #user enters a textfile
        with open(inputs_file) as file:
            entries = [i.strip("\n").split(" ") for i in file.readlines()]
            
        num_entries = len(entries)
        
        #define arrays to be saved by the predict method
        output = torch.zeros((num_entries, prediction_dim, 4, seq_len))
        ref_tensors = torch.zeros((num_entries, 4, seq_len))
        center_pos = []
            
        for i in range(len(entries)):
            chrome_num = entries[i][0]
            position = entries[i][-1]
            pos = int(position)
            center_pos.append(pos)
            
            
            #define inputs
            ref = encode_sequence(chrome_num, pos, genome, seq_len)
            ref_tensors[i] = ref
            
            

            
            #define model

            Predicted_chromatin_profiles = model(ref).detach().cpu()
            
            if difference: #if users want difference instead of l-f change
                
                if base_model and i == 0:

                    spec = importlib.util.spec_from_file_location(base_model, base_model_path)
                    model_class = getattr(spec.loader.load_module(), base_model) ##use this
                    base_model = model_class()
                    base_model_weights = base_weights
                
                elif i == 0:
                
                    base_model = Beluga().to(device) 
                    base_weights = '../CLIdata/deepsea.beluga.pth' 
                
                
                

                base_model.load_state_dict(torch.load(base_weights))
                base_model.eval()
                
                base_ref = base_model(ref.to(device)).cpu().T.unsqueeze(2)                         
                repeated_refs = base_ref.repeat(1, 4, seq_len)

                Predicted_chromatin_profiles = (log_unfold(repeated_refs, Predicted_chromatin_profiles[0]) - repeated_refs).unsqueeze(0)

            output[i] = Predicted_chromatin_profiles

            if add_tsv:
                col_names = get_col_labels(col_names) 
                Predicted_chromatin_profiles, ignore_indices = _reshape_prediction(ref, Predicted_chromatin_profiles, prediction_dim, seq_len)
                prediction = pd.DataFrame(Predicted_chromatin_profiles.detach().numpy(), columns = col_names)


                label = "_" + chrome_num + "_" + str(position) + "_"
                prediction.index = get_row_labels(chrome_num, int(position), ignore_indices, seq_len)

                prediction.to_csv("./newSaves/" + output_name + "." + chrome_num + "_" + position + ".tsv", 
                sep = "\t",
                float_format='%.3f',
                index = True,
                compression="gzip")
                
                
        prediction_dict = {"prediction": output, "reference": ref_tensors, "center_pos" : center_pos, "diff": difference}
        torch.save(prediction_dict , "./newSaves/" + output_name + ".pth")        
        
                

    else: #if user inputs an individual chromosome number and position
        pos = int(position)

        #define inputs
        ref = encode_sequence(chrome_num, pos, genome, seq_len)
        

        Predicted_chromatin_profiles = model(ref).detach().cpu()

        if difference:
            
            if base_model:
                spec = importlib.util.spec_from_file_location(base_model, base_model_path)
                model_class = getattr(spec.loader.load_module(), base_model) ##use this
                base_model = model_class()
                base_model_weights = base_weights
                
            else:
                
                base_model = Beluga().to(device) 
                base_weights = '../CLIdata/deepsea.beluga.pth' 
                
            base_model.load_state_dict(torch.load(base_weights))
            base_model.eval()
            base_ref = base_model(ref.to(device)).cpu().T.unsqueeze(2)               
            repeated_refs = base_ref.repeat(1, 4, seq_len)

            
            Predicted_chromatin_profiles = (log_unfold(repeated_refs, Predicted_chromatin_profiles[0]) - repeated_refs).unsqueeze(0)
            
   
        prediction_dict = {"prediction": Predicted_chromatin_profiles , "reference": ref, "center_pos" : pos, "diff": difference}
        torch.save(prediction_dict , "./newSaves/" + output_name + ".pth")
        
        
        if add_tsv:
            Predicted_chromatin_profiles, ignore_indices = _reshape_prediction(ref, Predicted_chromatin_profiles, prediction_dim, seq_len)
            col_names = get_col_labels(col_names)
            prediction = pd.DataFrame(Predicted_chromatin_profiles.detach().numpy(), columns = col_names)
           

            label = "_" + chrome_num + "_" + str(position) + "_"
            prediction.index = get_row_labels(chrome_num, pos, ignore_indices, seq_len)

            prediction.to_csv("./newSaves/" + output_name + ".tsv", 
               sep = "\t",
               float_format='%.3f',
               index = True,
               compression="gzip")
           


