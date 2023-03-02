# Overview

Multiplexer is a Python library and command line interface tool that enables users to develop and apply Multiplexer sequence models. A "Multiplexer model" is an augmented neural network that is trained from a "Base" model to provide fast, simultaneous predictions for a large set of input variations, such as all possible single nucleotide variations (SNVs) for a single sequence. Here, we provide the pre-trained BelugaMultiplexer, which predicts the effects of all possible SNVs of a single 2,000 base-pair sequence for 2,002 chromatin profiles. Additionally, we provide a Jupyter notebook that contains a template for custom Multiplexer model training and development.

The command line tool features two methods, **predict** and **plot**. With **predict**, users can quickly generate a DNA-sequence and make predictions with either the trained BelugaMultiplexer model or their own Multiplexer model, and with **plot**, users can create visualizations of their prediction outputs.


# Installation

### Packages

It is recommended that users install Python 3.6 or above before running the Multiplexer code.

Additionally, install [PyTorch](https://pytorch.org/get-started/locally/). If you are using NVIDIA GPU, be sure to check that your version of PyTorch is compatible with your GPU.

To use the Multiplexer command line tool locally, start by downloading the latest commits from the source repository with
```sh
git clone https://github.com/jzhoulab/Multiplexer.git  
```
Then, make sure all packages in the [requirements.txt](https://github.com/jzhoulab/Multiplexer/blob/master/requirements.txt) file are installed. The requirements.txt file contains a list of required packages that can be installed simultaneously with

```sh
pip install -r requirements.txt
```

### Command Line Data

Lastly, before running the command line, users need to download data files from Zenodo. It is recommended that users `cd` into the Multiplexer directory and then download the data with 

```sh
wget https://zenodo.org/record/7504998/files/CLIdata.zip?download=1
```
Alternatively users can use this [link](https://zenodo.org/record/7504998#.Y7ZCxuzMKrw) to download the data off the website and move the file into the Multiplexer directory. 

These files are initially zipped and can be unzipped with 

```sh
unzip CLIdata.zip?download=1
```
  
After these requirements are met, navigate to the CLI directory in Multiplexer via the command line and you are ready to start!


# Quick Start

### Single prediction
With **predict**, users can make a chromatin profile prediction with the BelugaMultiplexer model by specifying a chromosome value, a center position, and a output file name with the command:
```sh
python CLI.py predict --chromosome=<chromosome> --position=<pos> <output_name> 
```
\<chromosome> specifies which chromosome the input sequence is drawn from and should be given in the format 'chr[0-22,X,Y]' (e.g 'chr11'), \<pos> is an integer that specifies the position within the chromosome that the sequence is centered, and \<output_name> is the desired name of the output file that is saved to the './newSaves' directory.

For example, entering: 
```sh
python CLI.py predict --chromosome=chr11 --position=1000000 my_prediction 
```
Generates a 2,000 base-pair DNA sequence sampled from chr11 that is centered around position 1000000, passes the sequence through the BelugaMultiplexer model, and saves the prediction as part of a python dictionary. This dictionary can then be loaded in python by calling `torch.load('./newSaves/my_prediction.pth')` and using the `'prediction'` key. 

Note that the first run of **predict** will be comparatively slow as the `pyfasta` module is processing the genome object. However, subsequent runs will be notably faster. 
  
  
### Multiple predictions

In addition to making predictions one at a time, **predict** also allows users to input a tab separated text file and make multiple predictions all at once. In each row, input files should specify a chromosome as well as a position. For example:

```sh
chr3 825972
chr21 910413
chr14 1255421
...
```
Then, pass the file to the predict method using the command:
  
```sh
python CLI.py predict --inputs_file=my_input_file my_output_file
```
Where `my_input_file` is the relative path to the file containing model inputs.

**predict** will then make predictions for each value of the inputs file using BelugaMutliplexer and save the them into a single pickle file. To access the predictions, use `torch.load('./newSaves/my_output_file.pth')` to load in the prediction dictionary, and use the key `'predictions'` to retrieve the predicted values as a torch.tensor. 
 
  
### **Plotting Predictions**
After using **predict**, **plot** can take the saved output file and plot a 4x2000 heatmap of the predictions. On top of each position of the heatmap, the heatmap also displays the reference nucleotide.

To use **plot** in the command line, enter:

```sh
python CLI.py plot --plot_file=<input_file> <output_name> 
``` 
Where \<input_file> is the relative path to the Multiplexer predictions saved by **predict** and \<output_name> is the desired name for the saved plot. Note, plots and predictions will be saved by default into the `./newSaves' directory.
 
# Training Notebook

To help users train a custom SNV Multiplexer model, we provide a [Training.ipynb](https://github.com/jzhoulab/Multiplexer/tree/master/training#:~:text=34%20minutes%20ago-,Training.ipynb,-new) that contains starter code that can, with few adjustments, be adapted to train various models of custom input/output sizes. 

The training methods provided in the notebook are based on the code used to train the BelugaMultiplexer model but have been adjusted to accomodate models of varying dimensions. Given a custom base model, the notebook can be used to generate training data, perform forward and backward propogation, and save the trained model's parameters.

A second training notebook ['demoTraining.ipynb'](https://github.com/jzhoulab/Multiplexer/tree/master/training#:~:text=35%20minutes%20ago-,demoTraining.ipynb,-new) demonstrates custom Multiplexer training with a Base model that uses different input/output dimensions. When users train their own Multiplexer model, they should provide both a '.py' file that contains a Base model as well as a file containing trained Base model weights.

To use the command line methods with a custom trained Multiplexer, simply create a '.py' file that contains the Multiplexer model (remember to import torch and nn) and a file that contains the trained model parameters. 


# Predict

The full **predict** method is shown below and the full documentation can be found [here](https://github.com/jzhoulab/Multiplexer/blob/master/Documentation.md#documentation)


```sh
 python CLI.py predict ( --inputs_file=<filename> | --chromosome=<chr> --position=<pos>) <output_name> [--add_tsv]
[--diff] [(--modelname=<modelname> --modelpath=<modelp> --weights=<weights> --seqlen=<len> --predlen=<pred_len>)] 
[(--basename=<bmodel> --basemodelpath=<bmp> --baseweights=<bweights>)] [--colname=<cnames>] [--genome=<genome>]
```

  
### Predictions with BelugaMultiplexer
  
By default, **predict** makes predictions with the trained BelugaMultiplexer model. Instructions are shown in the [Quick Start](https://github.com/jzhoulab/Multiplexer#quick-start) section
  
Optionally, "--diff" and "--add_tsv" can be added to the end of the predict command. 
  
 `--diff` predicts the difference between the alternative prediction and the reference prediction rather than the log-fold change between the alternative and reference. 
 
 `--add_tsv` saves the predicted output as a 'tsv.gz' file in addition to the default pickle file. If a chromosome and position value are provided, the file will be saved as `output_name.tsv.gz`, and if an input file is given, multiple tsv files will be created and named in the format `output_name.chromosome_positionValue.tsv.gz`. 

For example, a prediction with `--diff` and `--add_tsv` can be made with:
  
```sh
python CLI.py predict --chromosome=chr11 --position=1000000 my_prediction --diff --add_tsv
```
This command saves the files `my_prediction.pth` and `my_prediction.tsv.gz` .

  
### Single Predictions with custom model

In addition to making predictions with the trained BelugaMutliplexer models, the command line tool also supports predicting and plotting with a custom trained DNA-sequence Multiplexer model. To use **predict** with a custom model, at a minimum, users must provide  `--modelname=<modelname>`, `--modelpath=<modelp>`, `--weights=<weights>`, `--seqlen=<len>`, and `--predlen=<pred_len>` in addition to either an inputs file or a chromosome and position pair. This corresponds with the name of the Multiplexer model, a relative path to a file containing the model, a relative path to a file containing the model weights, the length of the model input, and number of features predicted by the model (output dimension).
  
Example of custom model prediction:
```sh
python CLI.py predict chromosome=chr8 position=1000000 my_prediction --modelname=myMultiplexer 
  --modelpath=./myDirectory/mymodel.py  --weights=./myDirectory/modelweights.pth --seqlen=1000 --predlen=1002
```
The model myMultiplexer will make a prediction and save it to a file titled  `my_prediction.pth`. The input sequence for myMultiplexer has length 1,000 and the output has dimension 1,002.

  
### Output format
  
 For both single and multi-predictions, the output will be a python dictionary that is saved as a '.pth' file and can be accessed with the command `dictionary = torch.load('file_name.pth')`. The tensor containing the prediction can be accessed by loading in the dictionary and using the key `'prediction'`.

If a single prediction is made, the output tensor will have dimensions `[2002, 4, 2000]` - 2002 is the number of chromatin profiles, 4 represents the 4 basepairs (A,G,C,T), and 2000 represents each position in the sequence. If a file of inputs is provided, the output shape will be `[# of inputs, 2002, 4, 2000]`  where users can index along axis 0 to retrieve the desired prediction. For example, `[2, 2002, 4, 2000]` would correspond to the BelugaMultiplexer prediction for the 3rd input in the input file. 
  
To access a saved tsv file saved with the `--add_tsv` flag , it is suggested that users use `pd.read_csv('output_name.tsv', compression = 'gzip', sep = '\t', index_col = 0)` to load in a pandas dataframe with accurate row and column names. Column names contain the name of chromatin profiles such as "Chorion|DNase" and row names indicate the original chromosome, the position, and the basepair mutation such as "chrX:1000000 C>A".
  
# Plot
After **predict** is used and an output is saved, **plot** can be used to generate and save a heatmap of the saved prediction. The full menu of options is shown below:
  
```sh
CLI.py plot --plot_file=<plot_file> <output_name> [--file_index=<findex>] [--target_index=<tindex>] 
[--target_names=<tnames>][--ppr=<ppr>] [--figsize=<fsize>] [--output_format=<oformat>]
```
  
**plot** will create a heatmap showing the predicted mutation effects at all basepairs of the sequence: blue and red colors indicate negative and positive effects respectively. Specifically, the heatmap shows the log fold change of the prediction (log(ALT/(1-ALT))-log(REF/(1-REF)) (the default), or the difference (ALT-REF) if the --diff flag is used. Additionally, the reference sequence is shown on top of the heatmap, with darker color indicating more important bases (average mutation effects are more negative).

By default, the index of the chromatin profile with the smallest value is plotted, but the users can optionally specify a target index with the \<index> argument. Furthermore, the plot title will indicate which target index was plotted. Users can alternatively provide a file of target names with the `--target_names` command and the plot title will be retrieved from the file.

# Manuscript

A second repository containing the code to produce the results and figures from the manuscript can be found [here](https://github.com/jzhoulab/Multiplexer-Manuscript).

