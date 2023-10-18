import matplotlib.pyplot as plt
import pandas as pd
import collections
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pylab import *
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from matplotlib.patches import Patch


df = pd.read_csv("preprocessed_data/0628SPR_fft_rev5.csv").T
# df = pd.read_csv("preprocessed_data/0628spikes_cut.csv").T
print("df", df)


dataset = df.ffill()  # fill the nan using next row value
dataset = dataset.bfill()
X = dataset.iloc[:, :].values  # remove the first row where is the header

# this following part remove if no complex number
# # Convert the complex number strings to complex numbers
# X_complex = np.vectorize(np.complex)(X)
# X_real = np.concatenate((np.real(X_complex), np.imag(X_complex)), axis=1)
# X = X_real


X_scaled = TimeSeriesScalerMeanVariance(mu=0, std=1).fit_transform(X)  # standardization
X = X_scaled.squeeze()  # Reshape back to original shape

# df_label = pd.read_csv("entry3_hier_ward_euclid_spikes_cut/euclidean_k=2_cluster_labels.txt", header=None)
# df_label = pd.read_csv("entry1_kmeans_spikes_cut/kmeans_k=2_cluster_labels.txt", header=None)
# df_label = pd.read_csv("entry2_hier_avg_cos_spikes_cut/cosine_k=2_cluster_labels.txt", header=None)
df_label = pd.read_csv("entry6_hier_ward_euclid_fft_rev5/euclidean_k=16_cluster_labels.txt", header=None)
y = df_label.iloc[0:, 0].values  # labels
# target_clusters = [8, 11, 15, 16]  # Define target clusters to highlight
# target_clusters = [11]  # Define target clusters to highlight
target_clusters = [1,2,3,4,5,6,7,9,10,12,13,14]  # Define target clusters to highlight

# visualization
scatter_size = 30

# For PCA
# pca = PCA(n_components=2)
# embedding = pca.fit_transform(X)
# title = 'pca'

# For t-SNE
tsne = TSNE(n_components=2, random_state=10000)
embedding = tsne.fit_transform(X)
title = 'tsne'

# For UMAP
# reducer = umap.UMAP(n_components=2, random_state=10000)
# embedding = reducer.fit_transform(X)
# title = 'umap'

# Define extended list of colors
extended_colors = ['blue','green','red',  'skyblue', 'magenta',
                   'yellow', 'cyan', 'purple', 'brown', 'pink', 'orange', 'teal']

# Create color and alpha maps using dictionaries
color_map = {cluster: (extended_colors[i] if cluster in target_clusters else 'grey')
             for i, cluster in enumerate(target_clusters)}
alpha_map = {i+1: (1 if i+1 in target_clusters else 0.1) for i in range(max(y))}

# Assign colors and alphas to y using the maps
y_color = np.array([color_map.get(val, 'grey') for val in y])
y_alpha = np.array([alpha_map.get(val, 1) for val in y])  # 1 will be used for any unexpected labels


fig, ax = plt.subplots(figsize=(13, 12))
for cluster in set(y):
    mask = y == cluster
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=y_color[mask], alpha=y_alpha[mask], s=300, edgecolor='grey')

ax.set_title(title, fontsize=30)

ax.xaxis.set_tick_params(pad=15)
ax.yaxis.set_tick_params(pad=15)


# Create legend for the plot
legend_elements = [Patch(facecolor=color_map.get(cluster, 'grey'), edgecolor='k',
                         label=f'Cluster {cluster}') for cluster in target_clusters]
ax.legend(handles=legend_elements, loc='upper left',
# ax.legend(handles=legend_elements, loc='upper right',
          title='Clusters', title_fontsize='15', fontsize='16')

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)


plt.tight_layout()
plt.show()





