import pyfaidx 
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import pandas as pd

import sys
sys.path.append('../models')
from Beluga import Beluga



def get_data(input_file_path, file_index):
    """
    Retrieves data saved by the predict method
    
    Args:
        input_file_path : str
            The path to an input file saved from predict
            
        file_index : int
            The index of the inputs file to be plotted by the plot method
    
    """
    data = torch.load(input_file_path)
    
    #Raise exception if multiple input file is provided but an index is not
    if type(data['center_pos']) == list and file_index == None:
        raise Exception("If an input file contains multiple predictions, a file index must also be specified")
        
    
    i = 0 if file_index == None else int(file_index) - 1
    
    
    center_pos = data['center_pos'] if type(data['center_pos']) != list else data['center_pos'][i]
    return data["prediction"][i], data["reference"][i], center_pos, data["diff"]
   



def encode_sequence(chrome_num, pos):
    """
    Returns a sequence from chromosome chrome_num centered at position pos encoded
    as a 4xlength tensor
    
    Args:
        chrome_num : str
            The chromosome to be sample from (e.g chr11)
            
        position : int
            Center position of the chromosome to be sample from
            
        genome : fasta object
            The genome that sequences are drawn from
            
        length : int
            Length of input sequence
            
   Returns:
       seq_encoded : torch.tensor
           encoded version of the input sequence
        

    """
    seq = str(genome[chrome_num][pos-1000:pos+1000])

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




def plot_letters(axs, motifs, colors, ppr):
    """
    Given an axis and array of reference allele indices, plot letters on the heatmap
    
    Args:
        axs : Matplotlib axes
            Axes object to plot on
            
        Motifs : List
            A list of numbers representing the letters to be plotted
            
        Colors : (int,int,int) tuple
            Tuple representing the RGB value of the letter to be plotted
            
        ppr : int
            position-per-row is the number of positions plotted in each row of the output plit
    
    
    """
    scaled_font_size = round(12 / (ppr/200),2)
    for i in range(len(colors)):
        color = colors[i]

        if motifs[i] == 0:
            axs.text(i + 0.25,0,"A", fontsize = scaled_font_size, color = color, fontfamily = "monospace", fontweight = "bold")

        if motifs[i] == 1:
            axs.text(i + 0.25,0,"G", fontsize = scaled_font_size, color =  color, fontfamily = "monospace", fontweight = "bold")

        if motifs[i] == 2:
            axs.text(i + 0.25,0,"C", fontsize = scaled_font_size, color = color, fontfamily = "monospace", fontweight = "bold")

        if motifs[i] == 3:
            axs.text(i + 0.25,0,"T", fontsize = scaled_font_size, color = color, fontfamily = "monospace", fontweight = "bold")
            
            
            
def get_colors(array, diff = False):
    """
    Return the colors of the letters to be plotted on the graph
    
    Args:
        array : torch.tensor
            array containing the average values of the Multiplexer predictions made from the alternative sequences
            
        diff : bool
            whether the --diff option was used
            
    Returns:
        colors : list
            a list contain tuples of RGB values of colors
    
    
    """
    colors = []
    if diff:
        
        minval = torch.abs(array[array < 0]).max().item() #abs value of the min
        maxval = array.max().item()
        primary_color = 0.9
        for i in array:
            
            if i < 0:
                second_color = round(0.9 - (0.9)/(minval)*(torch.abs(i).item()), 2)
                color = (second_color, second_color, primary_color)
            elif i == 0:
                color = (0.9, 0.9, 0.9)
            elif i > 0:
                second_color = round(0.9 - (0.9)/(maxval)*(torch.abs(i).item()), 2)
                color = (primary_color, second_color, second_color)
                
            colors.append(color)

        
    else:
        minval = 0
        maxval = torch.abs(array[array < 0]).max().item()
        for i in array:
            if i < 0:
                color = round(0.9 - (0.9)/(maxval - minval)*(torch.abs(i).item() - minval), 2)
                
            else:
                color = 0.9
                
            colors.append( (color, color, color) )
         
    
    
    return colors
    
    
    

def create_plot(plot_index, plot_array, letters, colors, x_ticks, figname, target_names = None, ppr = None, figsize = None, output_format = None):
    """
    This method creates the plots and saves them as a pdf file
    
    Args:
        plot_arry : torch.tensor
            array to be plotted
            
        letters : list
            list of letters representing the reference allele to be plotted
            
        colors : list
            list of tuples representing the color of letters to be plotted
            
        x_ticks : lsit
            a list of x-axis labels
            
        figname : str
            name of plot
            
        ppr : int
            positions per row
            
        figsize : tuple
            size of plot
            
        output_format : str
            format of output plot (e.g 'pdf')
            
    
    """
    #define optional arguments
    ppr = 200 if ppr == None else ppr 
    figsize = (50,50) if figsize == None else figsize
    output_format = 'pdf' if output_format == None else output_format
    
    
    
    #calculate plot shapes and sizes
    total_plots = math.ceil(plot_array.shape[0]/ppr)
    remainder = plot_array.shape[0] % ppr #the number of positions to be plotted on the final plot
    
    
    
    min_val = plot_array.min()
    max_val = plot_array.max()

    
    fig = plt.figure(figsize = figsize)
    
    fig.subplots_adjust(hspace = 2)
    
    plot_dict = {}
    for i in range(1, total_plots + 1):
        plot_dict["ax" + str(i)] = fig.add_subplot(total_plots,1,i) 
        

    if remainder > 0:
        filler = torch.zeros((ppr - remainder,4))
        plot_array = torch.cat((plot_array, filler), axis = 0)
        x_ticks += [0 for i in range(ppr - remainder)]
    
    for j in range(1, total_plots + 1):
        
        plot_letters(plot_dict['ax' + str(j)], letters[(j-1)*ppr:j*ppr], colors[(j-1)*ppr:j*ppr], ppr)
        ret = sns.heatmap(plot_array[(j-1)*ppr:j*ppr].T, cbar=True, center = 0, vmin = min_val, vmax = max_val, ax = plot_dict['ax' + str(j)], cmap = 'RdBu_r')
        

        if j == 1: 
            if target_names:
                file = open(target_names)
                names = file.read().split("\n")
                title = names[plot_index]
 
            else:
                title = "Target Index Plotted: " + str(plot_index)
                
            
            mpl.rcParams['axes.titlesize'] = 40
            ret.set_title(title, y = 1.3)
            
                  
        ret.set_yticks([i + 0.5 for i in range(4)])
        ret.set_yticklabels(labels = ['A','G','C','T'], rotation =0)
        ret.set_xticks([i*5 + 0.5 for i in range(int(math.ceil(ppr/5)))])
        ret.set_xticklabels(x_ticks[(j-1)*ppr:j*ppr:5], fontsize = 10)
        
        if j == total_plots and remainder != 0:
            xticks = plot_dict['ax' + str(j)].xaxis.get_major_ticks()
            

            for i in range(int(math.ceil(remainder/5)), int(ppr/5) + 1):
                xticks[i].set_visible(False)
            

    plt.savefig("./newSaves/" + figname + "." + output_format, format = output_format)
    
    

def plot(input_file_path, output_name, ppr, file_index = None, target_names = None, user_index = None, figsize = None, output_format = None):
    """
    This method formats the data, sets up the coloring scheme, and determines the values to plot
    
    Args:
        input_file_path : str
            Which values to plot. This should be a filed saved from the predictions method
            
        output_name : str
            Name of file where the output plot is saved
            
        ppr : int
            positions per row
            
        file_index : int
            The index of a prediction tensor (saved by the predict method) to be plotted. This argument is required if users     
            provide an input file that contains the Multiplexer prediction for multiple inputs.
      
        user_index : int
            The index of the predicted feature to plot
            
        figsize : tuple (int,int)
            The size of the saved figured. Default = (50,50)
            
        output_format : str
            The output format of the plot e.g. 'pdf', 'png', etc.
    
    
    """
    figsize = (50,50) if figsize == None else (int(figsize.split(",")[0]), int(figsize.split(",")[1]))
    
    #call model on the inputs
    Predicted_chromatin_profiles, reference, position, diff = get_data(input_file_path, file_index)
    length = reference.shape[-1]
    ppr = 200 if ppr == None else int(ppr)

    
   
    ref_indices = []
    for i in range(length):
        if reference[:,i].sum() == 0: #in the case the reference nucleotide was 'n'
            ref_index = 0
        else:
            ref_index = torch.nonzero(reference[:,i]).item()

        ref_indices.append(ref_index)


    #get mean alternative predictions
    
  
    alt_predictions = Predicted_chromatin_profiles * (reference!=1)[ None,:,:] 
    alt_predictions = alt_predictions.sum(axis=1)/3
    alt_predictions = alt_predictions
   


    if user_index:
        plot_index = int(user_index)
        
    else:
        #get strongest index#
        max_arr = alt_predictions.min(axis = 1)
        plot_index = torch.argmin(max_arr[0]).item()



    plot_array = Predicted_chromatin_profiles[plot_index, :, :].T.detach()
    predictions = alt_predictions[plot_index, :]
    plot_array[torch.arange(length), ref_indices] = 0
    


    #get letters to plot
    letters = ref_indices #plot the letters
    x_ticks = [i for i in range(position - int(math.floor(length/2)) + 1, position + int(math.ceil(length/2)) + 1)] 
    letter_colors = get_colors(predictions, diff)
    



    create_plot(plot_index, plot_array.detach().cpu(), letters, letter_colors, x_ticks, output_name, target_names, ppr, figsize, output_format)

