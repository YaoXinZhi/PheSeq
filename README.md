# EMFAS
**Evidence-augmented Generative Model with Fine-grained weAk Labels (EMFAS) Model for Gene-Disease Association Discovery**

EMFAS is an abbreviation of "Evidence-augmented Generative Model with Fine-grained weAk Labels ". This is a Generative model with Bayesian framework. Please follow the below directions to run this model.

## Data preprocessing


### Data Collection
The **HeterogeneousData** folder contains both **Embedding data** and **p-value data** for three diseases. 

The text data in **HeterogeneousData/EmbeddingData/TextData** are downloaded from PubTator (https://www.ncbi.nlm.nih.gov/research/pubtator/). In the case when ones would like to collect all literature data related to an interested disease, please search the disease name in PubTator database and download all the Json/PubTator/BioC files.

The graph embedding **HeterogeneousData/EmbeddingData/GraphData** are downloaded from BioNEV (https://github.com/xiangyue9607/BioNEV). 

The download link of p-value data is recorded in **HeterogeneousData/P-ValueData/README.md**. The GWAS Summary data for AD are collected from GWAS Catalog (https://www.ebi.ac.uk/gwas/), and both transcriptome data for BC and methylation data for LC are collected from TCGA (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). For the disease under consideration, GWAS Summary data need be collected from resource like GWAS Catalog. Please be sure to include both gene site and p-value in the file.

**More information about collection and pre-processing of heterogeneous data can be found in "HeterogeneousData/README.md"**

## Run Model
 
    python src/run_model.py -ef [embedding_file] -sf [summary_file] --sl -lp [log_save_path] -lf [log_prefix] -mh [multi_hidden] --rs [random_seed] -ed [embedding_size] -lr [learning_rate] -hd [hidden_dim] -tt [train_time] -bs [batch_size] -pt [p_value_threshold]
    
**\[embedding_file]:** The embedding file.  
**\[summary_file]:** The summary file, file format is described in **"HeterogeneousData/EmbeddingData/README.md"**.  
**\[log_save_path]:** log_save_path.    
**\[log_prefix]:**  "predict", prefix of the log file name.  
**\[multi_hidden]:** False, Use more hidden layers.   
**\[random_seed]:** 126, random size.   
**\[embedding_size]:** 128, embedding size.  
**\[learning_rate]:** 5e-3, learning rate. 
**\[hidden_dim]:** 50, the dimension of hidden layer. 
**\[train_time]:** 100, training time. 
**\[batch_size]:** 128, batch size.  
**\[p_value_threshold]:** 5e-3, The threshold of p-value.  
