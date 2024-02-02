# src for result  
This folder contains the code used to process the results of the EMFAS model.  
EMFAS is an abbreviation of "Evidence-augmented Generative Model with Fine-grained weAk Labels ". This is a Generative model with Bayesian framework. Please follow the below directions to run this model.  

## Generate_report.py  
The code processes the EMFAS predictions result to generate user-readable report files that are used for subsequent analysis.   
 
    python Generate_report.py --log_path [log_path] --summary_path [summary_path] --output_path [output_path] --output_prefix [output_prefix] --diseases_source [diseases_source]  
    

**\[log_path]:**  path for EMFAS prediction result.  
**\[summary_path]:** path for summary file generated from Data_preprocess.ipynb.  
**\[output_path]:** path for output.   
**\[output_prefix]:** prefix for output.  

## benchmark_compare.ipynb
The code is a jupyter notebook code that compares the EMFAS prediction results with the existing knowledge of DISEASES database and plots simple cumulative curves and calculations of evaluation metrics such as MRR.  

## Gene-Evidence-GO-HPO_Statistics.ipynb
The code is a jupyter notebook code that counts the number of traceable textual evidence in the EFMAS prediction results and the number of various biological concepts, and draws a bar chart.  

