# src for data observation  
This folder contains the code used to observer the embedding and p-value data.  
EMFAS is an abbreviation of "Evidence-augmented Generative Model with Fine-grained weAk Labels ".  

## Embedding_cluster_observation.py   
This code is used to display the embedding clustering results and requires two files. 

    python Embedding_cluster_observation.py -cf  [cluster_file] -ef [embedding_file] -sp [save_path]   
    

**\[cluster_file]:**  a file containing the embedding name and the corresponding cluster name, tab-separated.  
**\[embedding_file]:** a file containing the embedding name and the corresponding embedding vector, tab-separated. 
**\[save_path]:** path for saving figure.   


## Embedding_p_observation.py. 
This code is used to observe the intrinsic association mechanism between the embedded data and the p-values.

    python Embedding_p_observation.py -ef  [embedding_file] -pf [p_value_file] -sp [save_path]   

**\[embedding_file]:**  A file contains the embedding name and the embedding vector, tab key separated.   
**\[p_value_file]:** A file contains the embedding name and the p-value, tab key delimited.  
**\[save_path]:** path for saving figure.   


