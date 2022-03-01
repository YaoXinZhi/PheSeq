# Data Preprocessing

## Text Embedding Calculation
This step is used to perform BERT-based embedding calculation on the gene literature files as well as biomedical concept annotations.


    python bag_embedding.py --input [INPUT_FILE]
                            --output [SAVE_FILE]
                            --max_len [MAX_LEN]
                            --max_bag_size [MAX_BAG_SIZE]
                            --embedding_size [EMBEDDING_SIZA]
                            --model [MODEL]
                            --use_cpu
                            --sigle_sentence_file [SINGLE_SENTENCE_FILE]
                            --text_norm 
                               
**\[INPUT_FILE]:** the literature file or biomedical concept file for embedding calculation.    
> Example files can be found in HeterogeneousData/EmbeddingData/TextData.   
> 1. **literature file format** (Tab separated)   
> GENE_LINE:    $GENE_Symbol    $Entrez_ID  $p-valie  
> $PMID_1 Sentence_1   {$Tag_1, Tag_2}  
> $PMID_2 Sentence_2   {$Tag_1, Tag_2}  
> 2. **biomedical concepts file format** (Tab separated)  
> $id_1 $Concept_1  
> $id_2 $Concept_2  

**\[SAVE_FILE]:** saved embedding file.  
**\[MAX_LEN]:** 500, max length for sentences.    
**\[MAX_BAG_SIZE]:** 20, maximum number of sentences pre gene for embedding calculation.  
**\[EMBEDDING_SIZA]:**  1024, embedding size.  
**\[MODEL]:** "base" for "dmis-lab/biobert-base-cased-v1.1" and "large" for "dmis-lab/biobert-large-cased-v1.1".  
**\[use_cpu]:** embedding calculation using only cpu or not.  
**\[SINGLE_SENTENCE_FILE]:** True for biomedical concept file format and False for literature file format.    
**\[text_Norm]:** Using text normalization or not.  

## Graph Embedding filter
This step is used to map the raw graph embedding data to genes and Entrez ID. The raw graph embedding data can be found in **HeterogeneousData/EmbeddingData/GraphData**. The gene-ensemble mapping file can be download in https://ftp.ncbi.nih.gov/gene/DATA/gene2ensembl.gz, and "gunzip .gz_file" is used to decompress this file.  

After data download, the following commed is used to pre-process graph embedding data.


    python Get_Entrez_struc2vec_embedding.py --gene2ensemble_file [gene2ensemble_file]
                                             --node_file [node_file]
                                             --embedding_file [embedding_file]
                                             --save_file [save_file]

**\[gene2ensemble_file]:** gene-ensemble id mapping file, which downloaded from NCBI.    
**\[node_file]:** nodes file of graph embedding data, which can be found in this floder.  
**\[embedding_file]:** raw graph embedding file, which can be found in this floder.    
**\[save_file]:** processed graph embedding file.  

## Embedding Merge with fixed weights
This step is used to merge the three types of embeddings with fixed weights.


    python Concept_embedding_merge.py --summary_file [summary_file]
                                      --concept_embedding_file [concept_embedding_file]
                                      --embedding_save_file [embedding_save_file]

**\[summary_file]:** Summary data, which including the sentence descriptions and p-value for each gene.    
> Example files can be found in HeterogeneousData/EmbeddingData/TextData.   
> 1. **Summary file format** (Tab separated)   
> GENE_LINE:    $GENE_Symbol    $Entrez_ID  $p-valie  
> $PMID_1 Sentence_1   {$Tag_1, Tag_2}  
> $PMID_2 Sentence_2   {$Tag_1, Tag_2}  

**\[concept_embedding_file]:** concept embedding file generated from **Text Embedding Calculation**.  
**\[embedding_save_file]:** merged embedding save file.    


    python Text_embedding_merge.py --text_embedding_file [text_embedding_file]
                                   --concept_embedding_file [concept_embedding_file]
                                   --embedding_save_file [embedding_save_file]

**\[text_embedding_file]:** text embedding file generated from **Text Embedding Calculation**.
**\[concept_embedding_file]:** concept embedding file generated from **Text Embedding Calculation**.  
**\[embedding_save_file]:** merged embedding save file.    



## GWAS Summary data Pre-processing
**GWAS_data_process.py** can be referenced for pre-processing of GWAS Summary data, including mapping SNP loci to Gene Symbol and Entrez ID, getting the most significant p-value for each gene.



