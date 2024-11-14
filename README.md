# ML Driven Targeting for HIV-1 Inhibition

## Quickstart

```markdown
# Cloning the repo
git clone https://github.com/epfl-ada/ada-2024-project-trendtrackers
# Navigate to the repo
cd <repo>
# Install all the dependencies
pip install -r requirements.txt
```
## Project Structure

```markdown
.
├── README.md                  # The following README :)
├── results.ipynb              # Initial analyses, clustering and visualisations
├── implementations.py         # Python file with helper functions
├── requirements.txt           # Python packages required to run the code
├── .gitignore                 # Git ignore file
```

## Abstract

HIV-1 continues to be a significant global health challenge, and developing effective inhibitors for this virus is a critical yet complex goal for both industry and academia. Our objective is to identify families of ligands with shared properties that inhibit HIV-1, facilitating the design of novel, targeted inhibitors. To achieve this, we plan to cluster these ligands based on their structural characteristics. Additionally, to efficiently evaluate the binding affinities of potential new ligands, we aim to use machine learning to predict their binding affinity to HIV-1. This in silico approach offers a faster and more cost-effective alternative to traditional experimental testing, accelerating the development of promising candidates.

## Research questions

- What are the most studied targets in academy and in industry ?
- Can we find families of ligand that inhibits HIV1 and what are their properties ?
- Can we predict the affinity of new ligands ?
- Are the affinity of ligands within the same clusters similar in values ?

## Methods

In this project, we aim to apply clustering techniques to explore the properties of compounds targeting HIV-1 proteins and to develop a predictive model for ligand binding affinity. Our analysis will leverage the [BindingDB dataset](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_202411_tsv.zip), which provides detailed information on the structural properties of proteins and their associated ligands, as well as binding affinity data. Before analysis, the dataset will be carefully preprocessed to address missing values and to select the most relevant features for our study.

### Clustering

We experimented with various approaches to numerically represent ligands, including generating Morgan fingerprints from SMILES strings with the [RDKit](https://www.rdkit.org/) library and utilising embeddings from the [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) model. These molecular structure representations are then processed using dimension reduction techniques like PCA, t-SNE and UMAP combined with KMeans to visualise and identify clusters effectively. 

### Next Steps

Our next steps will involve analysing the properties of each cluster, focusing on aspects such as molecular structure similarity, molecular weight, hydrogen bonding potential, and binding affinity, both visually and quantitatively. This analysis aims to uncover relationships between ligand structures and their key properties, helping us understand which structural features correlate with binding efficacy.

We will also assess the effectiveness of our clustering methods in forming distinct and meaningful groups that align with observed binding affinities.

In parallel, we plan to develop a predictive machine learning model to estimate binding affinity (inhibitory constant Ki) for a given ligand-protein pair. We will experiment with models such as random forests and neural networks to identify the best approach for accurate affinity prediction.

## Timeline

- Week 10: More visualisations on the properties and their correlations to check the similarity of the compounds in each cluster (affinity, hydrogen bonds, hydrophobicity, inhibitory effects etc).
- Week 11: Building, training and testing predictive ML model.
- Week 12: Test if a ligand family could target the same target with similar affinity values with the predictive model developed.
- Week 13 and 14: Improve results and work on the website to display our data story.

## Organisation

- Clustering and visualisation of the properties of clusters: Nithujaa, Leonardo, Emma
- Predictive Model: Antonin, Sathvik
- Story: Emma, Nithujaa, Leonardo, Antonin, Sathvik

## Questions for the TA:

- Are the current results of clustering promising or do we have to finetune our clustering process?
- As the data for HIV is quite small, should we find some external dataset for validating our model?
