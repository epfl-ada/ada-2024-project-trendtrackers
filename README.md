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
├── README.md                  # The following README :)
├── results.ipynb              # Initial analyses, clustering and visualisations
├── src/
    ├── implementations.py     # Python file with helper functions
├── test/                      # Test folder - using for saved plots (for now)
├── requirements.txt           # Python packages required to run the code
├── .gitignore                 # Git ignore file
```

## Abstract

HIV-1 remains a significant global health challenge, and developing effective inhibitors for this virus is a crucial yet complex goal for both industry and academia. Our objective is to identify families of ligands with shared properties that can inhibit HIV-1, helping the design of novel, targeted inhibitors. To achieve this, we plan to use cluster algorithms to separate the ligands based on their structural and chemical characteristics. Additionally, we aim to use machine learning to efficiently predict the binding affinities of new ligands and generate potential new candidates. This in silico approach could provide a faster, more cost-effective alternative to traditional experimental testing, speeding up the development of promising inhibitors.

## Research questions

- What are the most studied targets in academy and in industry ?
- Can we find families of ligand that inhibits HIV1 and what are their properties ?
- Can we predict the affinity of new ligands ?
- Are the affinity of ligands within the same clusters similar in values ?
- With the characteristics we obtain from the clusters, can we generate new ligands?

## Methods

In this project, we aim to apply clustering techniques to explore the properties of compounds targeting HIV-1 proteins and to develop a predictive model for ligand binding affinity. Our analysis will leverage the [BindingDB dataset](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/bind/downloads/BindingDB_All_202411_tsv.zip), which provides detailed information on the structural properties of proteins and their associated ligands, as well as binding affinity data. Before analysis, the dataset will be carefully preprocessed to address missing values and to select the most relevant features for our study.

### Clustering

We experimented with various approaches to numerically represent ligands, including generating Morgan fingerprints from SMILES strings with the [RDKit](https://www.rdkit.org/) library and utilising embeddings from the [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) model. These molecular structure representations are then processed using dimension reduction techniques like t-SNE and UMAP combined with KMeans to visualise and identify clusters effectively. 

### Next Steps

Our next steps will involve analysing the properties of each cluster, focusing on aspects such as molecular structure similarity, molecular weight, hydrogen bonding potential, and binding affinity, both visually and quantitatively. This analysis aims to uncover relationships between ligand structures and their key properties, helping us understand which structural features correlate with binding efficacy.

We will also assess the effectiveness of our clustering methods in forming distinct and meaningful groups that align with observed binding affinities.

In parallel, we plan to develop a predictive machine learning model to estimate binding affinity (inhibitory constant Ki) for a given ligand-protein pair. To encode proteins, we will leverage [ProteinBERT](https://github.com/nadavbra/protein_bert) to extract numerical features. We will experiment with models such as random forests and neural networks to identify the best approach for accurate affinity prediction.

If time permits, we would also explore reverse-engineering ligands by leveraging the prediction power of our machine learning model as a metric, combined with a new generative model. This approach involves generating the structures of optimal ligands with potentially high binding affinity for a given set of targets. The newly generated ligands can then be visualized using ChemDraw or the visualization tools available in [RDKit](https://www.rdkit.org/).

## Timeline

- Week 10: More visualisations on the properties and their correlations to check the similarity of the compounds in each cluster (affinity, hydrogen bonds, hydrophobicity, inhibitory effects etc).
- Week 11: Building, training and testing predictive ML model.
- Week 12:
     * Test if a ligand family could target the same target with similar affinity values with the predictive model developed.
     * Analyse the properties of each cluster (structure, molecular weight, h-bonds, ...)
- Week 13:
     * Setting up the website
     * Choose the visualisations to show
     * Begin the text for the datastory        
- Week 14:
     * Improve results and work on the website to display our data story.

## Organisation

- Clustering and visualisation of the properties of clusters: Nithujaa, Leonardo, Emma
- Predictive Model and Generative Model: Antonin, Sathvik
- Story: Emma, Nithujaa, Leonardo, Antonin, Sathvik

## Website

You can find our data story on our [Website](https://trendtrackers.github.io/).
