import numpy as np
import pandas as pd

from scipy.stats import entropy

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go

from sklearn.manifold import TSNE, trustworthiness
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from transformers import AutoTokenizer, AutoModel
import torch
import umap

def plot_sorted_count(value_counts, nrows, ncols, n_per_plot, counted_name):
    palette = sns.palettes.color_palette('colorblind',n_colors=len(value_counts)+1)
    fig, axes = plt.subplots(ncols=ncols,nrows=nrows, figsize=(15,15))

    categories = value_counts.index
    category_to_num = {category: i for i, category in enumerate(categories)}
    value_counts_num = value_counts.rename(index=category_to_num)

    for num, ax in enumerate(axes.flatten()):
        x = value_counts_num.iloc[n_per_plot*num:n_per_plot*(num+1)]
        sns.barplot(x=x.index,y=x.values,hue=x.index,legend=False, palette=palette[n_per_plot*num:n_per_plot*(num+1)],ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_yscale('log')
        up_lim = np.max(value_counts)*1.1
        ax.set_ylim((1,up_lim))
        ax.set_title(f' {n_per_plot*num}-{n_per_plot*(num+1) - 1}th {counted_name}')
        ax.set_xlabel('')

    categories = [f"{num}: {category}" for num, category in enumerate(categories[:320])]
    n_columns = 2 
    split_categories = [categories[i:i + len(categories)//n_columns] for i in range(0, len(categories), len(categories)//n_columns)]

    plt.figtext(0.5,-0.001,'Legend:', ha='center', va='top', fontsize=18, fontweight=800)

    y_position = -0.04 
    for col, column in enumerate(split_categories):
        plt.figtext(0.5 + (col - 0.5) * 0.5, y_position, '\n'.join(column), ha='center', va='top', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.suptitle('Sorted Value Counts of {counted_name}', fontweight='bold', fontsize=14, y=1.01)


def generate_fingerprints(smiles_list, radius=2, n_bits=1024):
    """Generates Morgan fingerprints for a list of SMILES strings using MorganGenerator."""
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)  
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((1,), dtype=int)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(None)
    
    return np.array([fp for fp in fingerprints if fp is not None])


def plot_elbow_curve(
    method, param1_list, param2_list, Ks, 
    original_data, param1_name='n_neighbors', 
    param2_name='min_dist', metric='euclidean'):
    """
    Function to plot elbow curves of dimension reduced data, for clustering analysis on embeddings.

    Parameters:
        method (str): Dimensionality reduction method ('umap' or 'tsne').
        original_data (array): Original embeddings for UMAP, SMILES list for t-SNE.
        param1_list (list): List of values for the first parameter (e.g., 'n_neighbors' or 'radius').
        param2_list (list): List of values for the second parameter (e.g., 'min_dist' or 'perplexity').
        Ks (range): Range of k values for KMeans clustering.
        param1_name (str): Name of the first parameter.
        param2_name (str): Name of the second parameter.
        metric (str): Distance metric to use (for UMAP only).

    Returns:
        tuple: results (list), scores (array), silhouette_scores (list)
    """
    
    nrows, ncols = len(param1_list), len(param2_list)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    sns.set_palette('deep', n_colors=nrows * ncols)
    
    results = [[[] for _ in range(ncols)] for _ in range(nrows)]
    scores = np.zeros((nrows, ncols))
    silhouette_scores = [[[] for _ in range(ncols)] for _ in range(nrows)]
    
    for idx1, param1 in enumerate(param1_list):
        print(f'{param1_name.capitalize()} = {param1}')
        
        if method == 'tsne':
            embeddings = generate_fingerprints(original_data, radius=param1)
        
        for idx2, param2 in enumerate(param2_list):
            print(f'{param2_name.capitalize()} = {param2}')
            ax = axes[idx1][idx2]
            
            if method == 'umap':
                model = umap.UMAP(n_components=3, n_neighbors=param1, min_dist=param2, metric=metric)
                result = model.fit_transform(original_data)
                scores[idx1][idx2] = trustworthiness(original_data, result, n_neighbors=param1)
                
            elif method == 'tsne':
                model = TSNE(n_components=3, perplexity=param2, init='random', learning_rate='auto', random_state=0)
                result = model.fit_transform(embeddings)
                scores[idx1][idx2] = model.kl_divergence_
            
            results[idx1][idx2][:] = result
            
            inertia = []
            s_score = []
            for k in Ks:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(result)
                inertia.append(kmeans.inertia_)
                s_score.append(silhouette_score(result, kmeans.labels_))
            silhouette_scores[idx1][idx2][:] = s_score
            
            ax.plot(Ks, inertia, 'o-', markersize=8, label=f'{param1_name}={param1}, {param2_name}={param2}', color=np.random.rand(3,))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.grid(True)

    plt.suptitle(f'Elbow Method for Optimal k with varying {param1_name} and {param2_name}', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()

    return results, scores, silhouette_scores


def plot_silhouette_scores(
    s_scores, Ks, param1_values, param2_values, param1_name, param2_name, 
    y_lim, color_palette="inferno", title="silhouette score for different k"
):
    """
    Function to plot silhouette scores for clustering analysis.

    Parameters:
        s_scores (list): A nested list of silhouette scores for each combination of parameters.
        Ks (range): Range of k values for KMeans clustering.
        param1_values (list): List of values for the first parameter (e.g., 'n_neighbors' or 'radius').
        param2_values (list): List of values for the second parameter (e.g., 'min_dist' or 'perplexity').
        param1_name (str): Name of the first parameter to display in the title (e.g., 'NN' or 'r').
        param2_name (str): Name of the second parameter to display in the legend (e.g., 'd' or 'p').
        y_lim (tuple): Y-axis limits for the silhouette scores.
        color_palette (str): Color palette for plotting.
        title (str): Title for the overall plot.
    """
    ncols = len(param1_values)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))
    axes = axes.flatten()
    palette = sns.color_palette(color_palette, n_colors=len(param2_values))
    
    for idx1, row in enumerate(s_scores):
        for idx2, s_score in enumerate(row):
            axes[idx1].plot(Ks, s_score, label=f'{param2_name}={param2_values[idx2]}', color=palette[idx2])
        axes[idx1].legend()
        axes[idx1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[idx1].set_xlabel('number of clusters (k)')
        axes[idx1].set_ylabel('silhouette score')
        axes[idx1].set_ylim(y_lim)
        axes[idx1].set_title(f'{param1_name}={param1_values[idx1]}')
        axes[idx1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def normalized_entropy(counts):
    k = len(counts) 
    probs = counts / counts.sum()  
    raw_entropy = entropy(probs)  
    max_entropy = np.log(k) 
    return raw_entropy / max_entropy


def plot_cluster_distributions(
    results, best_ks, param1_values, param2_values, param1_name, param2_name,
    palette='viridis', figsize=(22, 22), title="Clusters distribution"
):
    """
    Function to plot the distribution of clusters for different combinations of parameters.
    Uses normalized entropy as indicator metric

    Parameters:
        results (list): List of dimensionality-reduced results for each combination of parameters.
        best_ks (list): List of best k values for each parameter combination.
        param1_values (list): List of values for the first parameter (e.g., radiuses or n_neighbors).
        param2_values (list): List of values for the second parameter (e.g., perplexities or min_dists).
        param1_name (str): Name of the first parameter to display in the title.
        param2_name (str): Name of the second parameter to display in the title.
        palette (str): Color palette for plotting.
        figsize (tuple): Figure size for the plot.
        title (str): Title for the overall plot.
    """
    nrows, ncols = len(param1_values), len(param2_values)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    for idx1, param1 in enumerate(param1_values):
        for idx2, param2 in enumerate(param2_values):
            ax = axes[idx1][idx2]
            kmeans = KMeans(n_clusters=best_ks[idx1][idx2], random_state=42)
            kmeans.fit(results[idx1][idx2])

            # Add cluster labels to the DataFrame
            labels = pd.DataFrame(kmeans.labels_, columns=['labels'])
            cluster_counts = labels['labels'].value_counts()
            entropy_score = round(normalized_entropy(cluster_counts), 4)

            sns.countplot(x='labels', hue='labels', data=labels, palette=palette, ax=ax)
            ax.set_title(f'Distribution of Clusters ({param1_name}={param1} | {param2_name}={param2} | entropy score={entropy_score})')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Ligands')

            legend_labels = [f'{cluster} ({cluster_counts.get(cluster, 0)})' for cluster in range(best_ks[idx1][idx2])]
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10) for _ in range(best_ks[idx1][idx2])]
            ax.legend(handles, legend_labels, title='Cluster')

    plt.suptitle(title, fontweight='bold', fontsize=18, y=1.001)
    plt.tight_layout()
    plt.show()


def plot_3d_clusters(results, min_index, k, method_name):
    """
    Function to create a 3D scatter plot with clusters based on dimensionality reduction results.

    Parameters:
        results (list): List of dimensionality-reduced results for each parameter combination.
        min_index (tuple): Index of the optimal parameters in the results list.
        k (int): Number of clusters for KMeans.
        method_name (str): Name of the method (e.g., 'TSNE' or 'UMAP') for labeling axes.
    """
    # KMeans clustering on the selected result
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(results[min_index[0]][min_index[1]])
    df = pd.DataFrame(results[min_index[0]][min_index[1]], columns=[f'{method_name}1', f'{method_name}2', f'{method_name}3'])
    df['Cluster'] = kmeans.labels_

    # Create 3D scatter plot
    fig = go.Figure()
    scatter = go.Scatter3d(
        x=df[f'{method_name}1'],
        y=df[f'{method_name}2'],
        z=df[f'{method_name}3'],
        mode='markers',
        marker=dict(size=5, color=df['Cluster'], colorscale='Inferno', opacity=0.7)
    )
    fig.add_trace(scatter)

    # Customize layout
    fig.update_layout(
        title=f"{method_name} Clusters",
        scene=dict(
            xaxis_title=f'{method_name}1',
            yaxis_title=f'{method_name}2',
            zaxis_title=f'{method_name}3',
            xaxis=dict(showgrid=True, gridwidth=2, gridcolor='gray'),
            yaxis=dict(showgrid=True, gridwidth=2, gridcolor='gray'),
            zaxis=dict(showgrid=True, gridwidth=2, gridcolor='gray'),
        ),
        height=800, width=1200,
        template="plotly_white"
    )

    # Display the figure
    fig.show()

    return df
