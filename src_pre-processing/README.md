# Data Preprocessing

## Text Embedding Calculation
This step is used to perform BERT-based embedding calculation on the gene literature files as well as biomedical concept annotations.


    python bag_embedding.py --input [INPUT_FILE]
                            --output [SAVE_FILE]
                            --max_len [MAX_LEN]
                            --max_bag_size [MAX_BAG_SIZE]
                            --embedding_size [EMBEDDING_SIZA]
                            --model [MODEL]
                            --sigle_sentence_file [SINGLE_SENTENCE_FILE]
                            --text_norm [TEXT_NORM]
                               
**\[INPUT_FILE]:** input file, literature file or biomedical concept file.
> Example files can be found in HeterogeneousData/EmbeddingData/TextData. 
> 1. **literature file format** (Tab separated) 
> GENE_LINE:    $GENE_Symbol    $Entrez_ID  $p-valie
> $PMID_1 Sentence_1   {$Tag_1, Tag_2}
> $PMID_2 Sentence_2   {$Tag_1, Tag_2}
> 2. **biomedical concepts file format** (Tab separated)
> $id_1 $Concept_1
> $id_2 $Concept_2

**\[SAVE_FILE]:** 5e-8, The threshold of p-value;  
**\[MAX_LEN]:** 100, to record the result at round 100 in iteration;  
**\[MAX_BAG_SIZE]:** 80, freguency hyper-parameter, to ensure the stable output.
**\[EMBEDDING_SIZA]:**  100, the number of the iteration rounds;  
**\[MODEL]:** 100, the hyper-parameter;   
**\[SINGLE_SENTENCE_FILE]:** data/sorted_IGAP.txt, the result file from Synchronization Filter.   
**\[TEXT_NORM]:** generate/IGAP_Wilcoxon/, the outputfolder.  




## Graph Embedding filter


## Embedding Merge with fixed weights



