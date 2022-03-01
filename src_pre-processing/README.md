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


## Embedding Merge with fixed weights



