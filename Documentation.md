# Documentation

The command line tool follows the formatting outlined by docopt, a command-line interface description language. Further details on using docopt and be found [here](http://docopt.org/)





## Predicting

To make Multiplexer predictions from the command line, CLI.py uses:

```sh
    CLI.py predict ( --inputs_file=<filename> | --chromosome=<chr> --position=<pos>) <output_name> [--add_tsv] [--diff] 
    [(--modelname=<modelname> --modelpath=<modelp> --weights=<weights> --seqlen=<len> --predlen=<pred_len>)]
    [(--basename=<bmodel> --basemodelpath=<bmp> --baseweights=<bweights>)] [--colname=<cnames>] [--genome=<genome>]
```

* `<filename>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to an input file that contains chromosome and position values for prediction. An example of a formatted input file &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; can be found [here](https://github.com/jzhoulab/MultiplexerDev/blob/main/inputs.txt).
          
          
* `<chr>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The chromosome from which the input sequence is drawn. Takes the format `chr[0-22,X,Y]` (e.g. 'chr8'). If a `<filename>` is &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; not provided, a chromosome and position must be specified. 

* `<pos>` : **int** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The position within the specified chromosome that the input sequence is centered on. If a `<filename>` is not provided, a 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; chromosome and position must be specified. 

* `<output_name>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The name of the output file. **predict** automatically saves the prediction as '\<output_name>.pth'. 

* `--add_tsv` : **bool** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A flag that, if raised, indicates to additionally save a .tsv.gz file containing the Multiplexer predictions.

* `--diff` : **bool**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A flag that, if raised, indicates to predict the difference between the alternative and the reference predictions rather than the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; log-fold change between the alternative and reference predictions.

* `<modelname>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The name of a user's custom model class found within the user provided .py file. 

* `<modelp>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to a .py file containing a user's custom model.

* `<weights> ` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to a user's custom model's weights. The weights are loaded into an instance of the model with &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `model.load_state_dict(torch.load(<weights>))`.

* `<len> ` : **int** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The number of basepairs within the input sequence [Default = 2000].

* `<pred_len>` : **int** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The number of prediction features or categories made by the user's custom model (e.g BelugaMultiplexer predicts 2,002 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; chromatin profiles). 

* `<bmodel>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The name of a user's Base model class found within the user provided .py file. This is only required if user's provides a \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; custom model AND uses the `--diff` flag. 

* `<bmp>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to a .py file containing a user's Base model. Only required if `<bmodel>` is specified. 

* `<bweights>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to a user's Base model's weights. Only required if `<bmodel>` is specified. 

* `<cnames>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to a file containing custom column names for a '.tsv.gz' file. Each row should provide 1 column name (e.g a model &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; that predicts 1000 features would require a column names file  that contains 1000 rows)

* `<genome>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to a custom genome [Default = hg.19]. 


## Prediction Output
The predict method from the command line outputs a '.pth' file that contains a python dictionary that can be loaded in with `torch.load(<output_name>.pth)`. These `.pth` files will be saved into the `./newSaves/` directory. 

*  **The contents of the dictionary can be access with the following keys :**
    * "Prediction" : A torch.tensor that contains the saved Multiplexer prediction.
    * "reference" : A torch.tensor that contains the 1-hot encoding of the reference sequence.
    * "center_pos" : A int that represents the center value of the reference sequence.
    * "diff" : A boolean that is True if the --diff flag was used to make the prediction.
    

If the `--add_tsv` flag is used, an additional '.tsv.gz' file is saved. It is suggested that users load in the prediction as a pandas dataframe with `pd.read_csv('output_name.tsv', compression = 'gzip', sep = '\t', index_col = 0)` -  column names of the dataframe contain the name of chromatin profiles such as "Chorion|DNase" and row names indicate the chromosome, the position, and the basepair mutation such as "chrX:1000000 C>A". 


## Plotting
Within the CLI.pi file, the method to plot predictions is define as

```sh
  CLI.py plot --plot_file=<plot_file> <output_name> [--file_index=<findex>] [--target_index=<tindex>] [--ppr=<ppr>]
  [--figsize=<fsize>] [--output_format=<oformat>]
```

* `<plot_file>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The path to an input file containing a prediction to be plotted. Input files must be created from the **predict** method.  

* `<output_name>` : **str** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The name of the file the output plot is saved in. By default, **plot** saves an output plot titled '<output_name>.pdf'.

*  `<findex>` : **int** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; When an file that contains multiple chromosome and position values is given to **predict**, all predictions are saved in a single &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tensor. \<findex\> specifies which prediction within the tensor is plotted.

* `<tindex>` : **int** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The index of the feature to be plotted. By default, the index of the feature with the largest value is plotted.

* `<ppr>` : **int** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The number of positions plotted in each row. The total rows plotted is equal to `ceiling(total number of positions/<ppr>)`. 

* `<fsize>` : **int, int** \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The size of the output figure given as two int values seperated by a comma, such as `40,40` [Default = (50,50)].

* `<oformat>` : **str**  \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The format of the file that contains the output plot [Default = 'Pdf']. The full list of formats can be found [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
