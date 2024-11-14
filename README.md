# Title: to find

## [OPTIONAL] create conda environment
conda create -y -n ada python=3.11 scipy pandas numpy matplotlib=3.7.2
## install requirements
pip install -r pip_requirements.txt

## Abstract

HIV-1 remains a major global health challenge, and developing effective inhibitors for this virus is a critical yet complex goal for both industry and academia. Our objective is to identify families of ligands with shared properties that inhibit HIV-1, facilitating the design of novel, targeted inhibitors. To achieve this, we plan to cluster these ligands based on their structural characteristics. Additionally, to efficiently evaluate the binding affinities of potential new ligands, we aim to use machine learning to predict their binding affinity to HIV-1. This in silico approach offers a faster and more cost-effective alternative to traditional experimental testing, accelerating the development of promising candidates.

## Research questions

- What are the most studied targets in academy and in industry ?
- Can we find families of ligand that inhibits HIV1 and what are their properties ?
- Can we predict the affinity of new ligands ?
- Are the affinity of ligands within the same clusters similar in values ?

## Methods

In this project, we will apply clustering methods to analyze specific properties of compounds targeting HIV-1 proteins and develop a predictive model for ligand binding affinity to a protein target. The analysis will utilize the [BindingDB dataset](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_202411_tsv.zip), which provides detailed information on the structural properties of proteins and their associated ligands, as well as binding affinity data. Before analysis, the dataset will be thoroughly preprocessed, including handling missing values and selecting relevant columns for analysis.

CLustering:
Presently, we experimented with various approaches, such as using Morgan fingerprints generated from 'Ligand SMILES' using the [RDKit](https://www.rdkit.org/) library and employing embedding models from the [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) library. These molecular structure representations are then processed using dimension reduction techniques like PCA, t-SNE, and UMAP combined with KMEANS to visualize and identify clusters effectively. 

Visualization:
The next step would be to analyse each cluster's properties, such as molecular structure similarity, molecular weight, hydrogen bonding potential, and binding affinity both visually and quantitatively.

This approach aims to uncover relationships between ligand structures and their properties. The choice of methods will be evaluated based on their ability to create distinct and meaningful clusters that correlate with observed binding affinities.

In parallel, a predictive machine learning would be developed in order to predict the binding affinity (inhibitory constant Ki) given a ligand and a protein. We would use a random forests, neural network etc.

## Timeline

- Week 10: More Visualizations on the properties and their correlations to check the similarity of the compounds in each cluster (affinity, hydrogen bonds, hydrophobicity, inhibitory effects, .... ).
- Week 11: Predictive ML model.
- Week 12: Test if a ligand family could target the same target with similar affinity values with the predictive model developed.
- Week 13 and 14: Create the Website to show our story.

## Organisation

- clustering and visualization of the properties of clusters: Nithujaa, Leonardo, Emma
- Prediction: Antonin, Sathvik
- Story: Emma, Nithujaa, Leonardo, Antonin, Sathvik

## Questions for the TA:

- Are the current results of clustering promising or do we have to finetune our clustering process?

