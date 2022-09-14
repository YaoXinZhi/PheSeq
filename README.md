# EMFAS
**Evidence-augmented Generative Model with Fine-grained weAk Labels (EMFAS) Model for Gene-Disease Association Discovery**

EMFAS is an abbreviation of "Evidence-augmented Generative Model with Fine-grained weAk Labels ". This is a Generative model with Bayesian framework. Please follow the below directions to run this model.

## Environment Configuration  
EMFAS have been tested using python3.7 on Ubuntu 21.04 and uses the following main dependencies on a CPU and NVIDIA GeForce RTX 3090Ti GPU:  

 
    torch==1.7.1
    sympy==1.8.0
    scipy==1.1.2
    transformers==4.10.2
    numpy==1.19.5
    spacy==2.3.5
    scikit-learn==0.20.0
    
Other dependency packages can be found in the **requeirement.txt**, and batch installed with the following commend line.  
 
    pip3 install -r requirements.txt


## Data preprocessing

### Pre-computed embedding for 32 Pan-Cancers in TCGA  
To facilitate the EMFAS implementation for more disease cases, pre-processed text and embedding data for 32 types of Pan-Cancers in TCGA database are offered in http://lit-evi.hzau.edu.cn/Bayes/more-diseases. The data include rich-annotated sentence evidence and pre-computed embedding for each gene.  

For these 32 cancers, the pre-computed embedding data can be downloaded directly, and the EMFAS model can be used to intergrate the sequence analysis data already developed by the user with embedding data.  


### Data Collection
The **HeterogeneousData** folder contains both **Embedding data** and **p-value data** for three diseases. 

The text data in **HeterogeneousData/EmbeddingData/TextData** are downloaded from PubTator (https://www.ncbi.nlm.nih.gov/research/pubtator/). In the case when ones would like to collect all literature data related to an interest disease, please search the disease name in PubTator database and download all the Json/PubTator/BioC files.

The graph embedding **HeterogeneousData/EmbeddingData/GraphData** are downloaded from BioNEV (https://github.com/xiangyue9607/BioNEV). 

The download link of p-value data is recorded in **HeterogeneousData/P-ValueData/README.md**. The GWAS Summary data for AD are collected from GWAS Catalog (https://www.ebi.ac.uk/gwas/), and both transcriptome data for BC and methylation data for LC are collected from TCGA (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). For the disease under consideration, GWAS Summary data need be collected from resource like GWAS Catalog. Please be sure to include both gene site and p-value in the file.

**More information about collection and pre-processing of heterogeneous data can be found in "HeterogeneousData/README.md"**

### Training with your own data  
If you want to use the EMFAS model for your own heterogeneous data -- a set of embedding data and a set of p-value data, then the two files need to be constructed.  
**1. Summary data**  
> Summary data, which including the sentence descriptions and p-value for each gene.  
> Example files can be found in **HeterogeneousData/EmbeddingData/TextData**.  
> The Summary file format as flowers (Tab separated):  
>> GENE_LINE:    $GENE_Symbol    $Entrez_ID  $p-valie  
>> $PMID_1 Sentence_1   {$Tag_1, Tag_2}  
>> $PMID_2 Sentence_2   {$Tag_1, Tag_2}   

**2. Embedding data**
> Embedding data, it can be derived from different representation learning methods, such as Graph embedding, text embedding.  
> Embedding files for mat as flowers (Tab separated):  
>> $Entrez_ID_1 $embedding_vec_1  
>> $Entrez_ID_2 $embedding_vec_2  

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


    python src/Dynamic EMFAS.py -mf [summary file] -ef [embedding file] -ds [data size] -ed [embedding size] -mh [multi hidden] -hs [hidden dim] -zd [z dim] -bs [batch size] -tt [train time] -lr [learning rate] -mr [mu learning rate] -sr [sigma learning rate] -rs [random seed] -ga [gradient accumulation] -sl [save log] -lp [log save path] -lf [log save prefix] -uf [use fake data]
    

**\[summary file]:** The summary file, file format is described in ”HeterogeneousData /Embed-dingData /README.md”.
**\[embedding file]:** The embedding file.
**\[data size]:** The data size (genotype-phenotype association count) of the input data.
**\[embedding size]:** The embedding dimension of the embedding data.
**\[multi hidden]:** Use more hidden layers.
**\[hidden dim]:** The hidden layer dimension of the deep learning module.
**\[z dim]:** The dimension of the latent variable Z.
**\v[log save path]:** The folder used to save the log file.
**\[log prefix]:** The name prefix of the log file.
**\[batch size]:** Data count of each mini-batch.
**\[train time]:** The training epoch.
**\[batch size]:** Batch size.
**\[learning rate]:** The learning rate.
**\[mu learning rate]:** The learning rate for μ-related part in the deep learning module.
**\[sigma learning rate]:** The learning rate for σ-related part in the deep learning module.
**\[random seed]:** Random size.
**\[gradient accumulation]:** Using gradient accumulation strategy in model training or not.
**\[save log]:** Save the log file or not.
**\[log save path]:** The folder for saving the log file.
**\[log save prefix]:** The name prefix for the saved log file.
**\[use fake data]:** True, using test file or not.



## Evidence-argumented pathological network visualization  

    python src_for_result/Pathological_evidence_network_visualization.py --report_file [report_file] -evidence_file [evidence_file] --save_file [save_file] --topn [topn] --tax_id [tax_id]  --sequence_analysis_gene_file [sequence_analysis_gene_file] --add_STRING [add_STRING] --filter_keyword [filter_keyword]
    
**\[report_file]:** report file generated by Generate_reports.py.     
**\[evidence_file]:** evidence file. It can be downloaded from http://lit-evi.hzau.edu.cn/Bayes/more-diseases.    
**\[save_file]:** saved pathological network visualization result, the suffix must be ".html".  
**\[topn]:** Top-n genes for visualization.   
**\[tax_id]:** tax id for species of interest, i.e. 9606 for Homo Sapiens.   
**\[sequence_analysis_gene_file]:** Significant gene file from sequence analysis, containing one column, each Entrez id per row.   
**\[add_STRING]:** add edges of gene-gene interaction from STRING database, default: False.   
**\[filter_keyword]:** keyword used to filter evidence example in pathological network visualization.  


