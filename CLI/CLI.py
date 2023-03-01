from docopt import docopt
from predict import *
from plot import *





usage = '''
Beluga Multiplexer CLI

Usage:
    CLI.py predict ( --inputs_file=<filename> | --chromosome=<chr> --position=<pos>) <output_name> [--add_tsv] [--diff] [(--modelname=<modelname> --modelpath=<modelp> --weights=<weights> --seqlen=<len> --predlen=<pred_len>)] [(--basename=<bmodel> --basemodelpath=<bmp> --baseweights=<bweights>)] [--colname=<cnames>] [--genome=<genome>] 
    CLI.py plot --plot_file=<plot_file> <output_name> [--file_index=<findex>] [--target_index=<tindex>] [--target_names=<tnames>][--ppr=<ppr>] [--figsize=<fsize>] [--output_format=<oformat>]
    CLI.py --help

Options:
    --inputs_file=<filename>   <filename> is the name of a file that contains multiple inputs           
    --chromosome=<chr>         <chr> is the chromosome (e.g chr11)
    --position=<pos>           <pos> is the center position of the sequence
    --add_tsv                  Saves the prediction in a tsv format in addition to a pickle file
    --diff                     Predict the difference between alt. and ref. allele instead of the log-fold change
    --modelname=<modelname>*   <modelname> is the name of the user's multiplexer model within the .py file
    --modelpath=<modelp>*      <modelp> is the path to the file containing the multiplexer model
    --weights=<mweight>*       <mweights> is path to the multiplexer model weights [default = BelugaMultiplexer weights]
    --seqlen=<len>*            <len> is the length of base-pair sequence [default = 2000]
    --pred_len=<pred_len>*     <pred_len> is the number of predictions made by the user's model [default = 2002]
    --basename=<bname>**       <bname> is the name of the user's base model within the .py file
    --basemodelpath=<bmp>**    <bmp> is the path to the file containing the base model
    --baseweights=<bweights>** <bweights> is path to model weights [default = Beluga weights]
    --colname=<cnames>         <cnames> is a file containing column names
    --genome=<genome>          <genome> is a path to user's custom genome [default = hg19.fa]
    --plot_file=<plot_file>    <plot_file> is a '.pth' file to be used for plotting
    --multi_file=<multi_file>  <multi_file> is a '.pth' file to be used for plotting that contains multiple predictions
    --file_index=<findex>      <findex> represents the row index of a multiple inputs file (<filename>) to be plotted
    --target_index=<tindex>    <tindex> which target index to plot [default = index of max chromatin profile prediction]
    --target_names=<tnames>     <tnames> file of target names to be plotted
    --ppr=<ppr>                <ppr> is positions per plot, the number of positions to be plotted on each row of the plot
    --figsize=<fsize>          <fsize> is the figure size in the format int,int [default = (50,50)]
    --output_foramt=<oformat>  <oformat> is the format the output file is saved
    --help                     show this screen
    
    *If users input a custom model, they must include all optional arguments denoted with a *
    **If users want to use --diff with a custom model, they must inlucde all optional arguments denoted with a ** 

'''
args = docopt(usage)



if args['predict']:
    predict(inputs_file = args['--inputs_file'], chrome_num = args['--chromosome'], position = args['--position'], output_name = args['<output_name>'], add_tsv = args['--add_tsv'], difference = args['--diff'], predict_model = args['--modelname'], predict_model_path = args['--modelpath'], predict_weights = args['--weights'], seq_len = args['--seqlen'], prediction_dim = args['--predlen'], genome = args['--genome'], base_model = args['--basename'], base_model_path = args['--basemodelpath'] ,base_weights = args['--baseweights'], col_names = args['--colname'])
    

    
        
if args['plot']:        
    plot(input_file_path = args['--plot_file'], output_name = args['<output_name>'], file_index = args['--file_index'], user_index = args['--target_index'], target_names=args['--target_names'], ppr = args['--ppr'], figsize = args['--figsize'], output_format = args['--output_format'])

    
if args['--help']:
    print(usage)

