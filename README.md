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

Note:
- Make sure to have the zip of the BindingDB dataset ['BindingDB_All_202409_tsv.zip'](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_202409_tsv.zip) in your repository.
- Avoid running the jupyter notebook since it takes some time to run (~20 min with decent computer). All visualization should already be loaded.
- We recommend using VSCode. The library Plotly is used to create 3D interactive plots, but for some reason, these plots do not render when using Jupyter online. However, they do render properly in VSCode. Other softwares have not been tested. Regardless, saved images of the plots are displayed in the notebook if you are not using VSCode.

## Project Structure

```markdown
.
├── README.md                        # The following README :)
├── results.ipynb                    # All the analyses, clustering, predictive modeling, visualisations
├── labeled_data/                    # Preprocessed data for ease of importing
├── src/
    ├── implementations.py           # Python file with helper functions
    ├── model_implementations.py     # Python file for affinity predictive model implementations
├── plots/                           # Using for saved plots
├── requirements.txt                 # Python packages required to run the code
├── .gitignore                       # Git ignore file
```

## Abstract

HIV-1 remains a significant global health challenge, and developing effective inhibitors for this virus is a crucial yet complex goal for both industry and academia. Our objective is to identify families of ligands with shared properties that can inhibit HIV-1, helping the design of novel, targeted inhibitors. To achieve this, we plan to use cluster algorithms to separate the ligands based on their structural and chemical characteristics. Additionally, we aim to use machine learning to efficiently predict the binding affinities of new ligands and generate potential new candidates. This in silico approach could provide a faster, more cost-effective alternative to traditional experimental testing, speeding up the development of promising inhibitors.

## Research questions

- What are the most studied targets in academy and in industry?
- Can we find families of ligand that inhibits HIV1 and what are their properties?
- What are the different properties of each cluster and how they differ from each other?
- Can we find characteristics which make it a good inhibitor?
- Can we predict the affinity of new ligands?

## Methods

In this project, we aim to apply clustering techniques to explore the properties of compounds targeting HIV-1 proteins and to develop a predictive model for ligand binding affinity. Our analysis will leverage the [BindingDB dataset](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_202411_tsv.zip), which provides detailed information on the structural properties of proteins and their associated ligands, as well as binding affinity data. Before analysis, the dataset will be carefully preprocessed to address missing values and to select the most relevant features for our study.

### Clustering

We experimented with various approaches to numerically represent ligands, including generating Morgan fingerprints from SMILES strings with the [RDKit](https://www.rdkit.org/) library and utilising embeddings from the [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) model. These molecular structure representations are then processed using dimension reduction techniques like t-SNE and UMAP combined with KMeans to visualise and identify clusters effectively. 

### Analysis

We analysed the properties of each cluster, focusing on aspects such as molecular structure similarity, molecular weight, hydrogen bonding potential, and binding affinity, both visually and quantitatively. This analysis helped us uncover relationships between ligand structures and their key properties, helping us understand which structural features correlate with binding efficacy.

### Predictive Model

We trained predictive machine learning models to estimate binding affinity (inhibitory constant Ki) for a given ligand for Gag-Pol Polyprotein. To encode proteins, we leveraged [ProteinBERT](https://github.com/nadavbra/protein_bert) to extract numerical features. We experimented with different models such as random forests and neural networks to identify the best approach for accurate affinity prediction.

## Contributions

- Clustering: Antonin, Sathvik, Nithujaa, Leonardo, Emma
- Analysis of the clusters: Nithujaa, Leonardo, Emma
- Predictive Model: Antonin, Sathvik
- Story: Emma, Nithujaa, Leonardo, Antonin, Sathvik
- Website: Sathvik

## Website

You can find our data story on our [Website](https://trendtrackers.github.io/).
