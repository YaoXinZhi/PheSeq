
# Data Collection
The **HeterogeneousData** folder contains both **Embedding data** and **p-value data** for three diseases. 

The text data in **HeterogeneousData/EmbeddingData/TextData** are downloaded from PubTator (https://www.ncbi.nlm.nih.gov/research/pubtator/). In the case when ones would like to collect all literature data related to an interested disease, please search the disease name in PubTator database and download all the Json/PubTator/BioC files.

The graph embedding **HeterogeneousData/EmbeddingData/GraphData** are downloaded from BioNEV (https://github.com/xiangyue9607/BioNEV). 

The download link of p-value data is recorded in **HeterogeneousData/P-ValueData/README.md**. The GWAS Summary data for AD are collected from GWAS Catalog (https://www.ebi.ac.uk/gwas/), and both transcriptome data for BC and methylation data for LC are collected from TCGA (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). For the disease under consideration, GWAS Summary data need be collected from resource like GWAS Catalog. Please be sure to include both gene site and p-value in the file.

# EMFAS
**Evidence-augmented Generative Model with Fine-grained weAk Labels (EMFAS) Model for Gene-Disease Association Discovery**

EMFAS is an abbreviation of "Evidence-augmented Generative Model with Fine-grained weAk Labels ". This is a Generative model with Bayesian framework. Please follow the below directions to run this model.

## Data preprocessing


## Run Model
 
    python src/run_model.py -ef [embedding_file] -sf [summary_file] --sl -lp [log_save_path] -lf [log_prefix] -mh [multi_hidden] --rs [random_seed] -ed [embedding_size] -lr [learning_rate] -hd [hidden_dim] -tt [train_time] -bs [batch_size] -pt [p_value_threshold]
    
**\[initlambda]:** 240, the hyper-parameter;  
**\[threshold]:** 5e-8, The threshold of p-value;  
**\[times]:** 100, to record the result at round 100 in iteration;  
**\[filter_count]:** 80, freguency hyper-parameter, to ensure the stable output.
**\[rounders]:**  100, the number of the iteration rounds;  
**\[hidden_factors]:** 100, the hyper-parameter;   
**\[inputfile]:** data/sorted_IGAP.txt, the result file from Synchronization Filter.   
**\[outputfolder]:** generate/IGAP_Wilcoxon/, the outputfolder.  


