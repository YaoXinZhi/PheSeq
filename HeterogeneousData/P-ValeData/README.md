# p-value Data Collection

## GWAS data for AD (or any disease)

When **GWAS data** for interest disease need be collected, please search the disease in **GWAS catalog** and download corresponding **GWAS Summary data**. Afterwards, beedtools can be used to map SNP loci to genes, thus corresponding genes to p-values.


## p-value data for cancer

In the case when ones would to collect the **p-value for cancer**, please search the disease in **TCGA database** and download corresponding files. The downloaded files needs to contain the p-value and Entrez ID for each disease-related gene, note that **both significant (less than the threshold) and insignificant (greater than threshold) p-values need to be included**.

# Data Preprocessing

After gene-p-value data is collected, the NCBI gene-entrez mapping file (https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz) need be used for converting gene symbol to Entrez ID.

The processed p-value data needs to have three columns separated by Tab keys, including **GeneSymbol, EntrezID and p-value**, the file format is same as the **P-ValueData/BC/BC.symbol_entrez.p.tsv**.


