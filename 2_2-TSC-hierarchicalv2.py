import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score

from tslearn.utils import to_time_series_dataset, to_time_series
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.ticker import FuncFormatter


def three_decimal(x, pos):  # formats a number to three decimal places
    return f'{x:.3f}'


# Define the tick formatter function
def subtract_40(x, pos):
    return f'{int(0.1*x + 40)}'


# dendrogram
def condensed_distance_matrix(distance_matrix):
    iu1 = np.triu_indices(len(distance_matrix), 1)
    return distance_matrix[iu1]


infile = "preprocessed_data/0628SPR_fft_rev5.csv"
# infile = "preprocessed_data/0628SPR_fft_coef_5.csv"
outfile = "entry6_hier_ward_euclid_fft_rev5"

# distance_flag = "cosine"  # or "euclidean"
distance_flag = "euclidean"  # or "euclidean"

k = list(range(2, 19))
# k = [12, 13, 14, 15, 16]
main_info = distance_flag + ""

df = pd.read_csv(infile)
df = df.T.ffill()  # transpose and fill the nan using next row value
indata = "all"

df2np = df.to_numpy()
print("shape", df2np.shape, df2np)
print(df2np.shape)
df2np_reshape = np.reshape(df2np, (2000, df2np.shape[1], 1))  # (samples, data len, 1)

seed = 1000
X_train = df2np_reshape
print("shape of training data", X_train.shape)

X_train_complex = np.vectorize(np.complex)(X_train)  # deal with complex number
X_train_real = np.concatenate((np.real(X_train_complex),
                               np.imag(X_train_complex)), axis=1)  # Concatenate the real and imaginary parts
X_train = X_train_real

X_train = TimeSeriesScalerMeanVariance(mu=0, std=1).fit_transform(X_train)
X_train = TimeSeriesResampler(sz=400).fit_transform(X_train)
print("X_train", X_train)
print("After shape of training data", X_train.shape)

# save the coef values
X_train_reshaped = X_train.reshape(2000, 400)
X_train_save = pd.DataFrame(X_train_reshaped)
X_train_save.to_csv('coef_X_train.csv', index=False)

os.makedirs(outfile, exist_ok=True)
# recursive_function()
summary_sil_scores = []
figs_per_row = 3


# Prepare the data
X_train_2d = X_train.reshape(X_train.shape[0], -1)

if distance_flag == "cosine":
    distance_matrix = cosine_distances(X_train_2d)
    affinity = 'precomputed'
    linkage_method = 'average'
elif distance_flag == "euclidean":
    distance_matrix = X_train_2d  # we will pass the original data directly
    affinity = 'euclidean'
    linkage_method = 'ward'

# Generate dendrogram
if distance_flag == "cosine":
    dist_condensed = condensed_distance_matrix(distance_matrix)
    linked = linkage(dist_condensed, linkage_method)
elif distance_flag == 'euclidean':
    linked = linkage(X_train_2d, linkage_method)

# Loop over the number of clusters
for n_clstr in k:
    print(f"{distance_flag.capitalize()} distance Agglomerative Clustering")

    # clustering
    model = AgglomerativeClustering(n_clusters=n_clstr, affinity=affinity, linkage=linkage_method)
    y_pred = model.fit_predict(distance_matrix)

    # save the clustering at each stage
    n_rows = (n_clstr - 1) // figs_per_row + 1
    fig_height = 6 * n_rows
    plt.figure(dpi=500, figsize=(24, fig_height))

    # count labels
    counts = Counter(y_pred)

    for yi in range(n_clstr):
        ax = plt.subplot(n_rows, figs_per_row, yi + 1)
        cluster_data = X_train[y_pred == yi]
        for xx in cluster_data:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        centroid = np.mean(cluster_data, axis=0)
        plt.plot(centroid.ravel(), "r-")

        # Add cluster count and label to plot
        plt.text(0.5, 0.95, f'Cluster {yi + 1} (Count: {counts[yi]})',
                 transform=plt.gca().transAxes, fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.tight_layout()

    # Calculate silhouette score and append to list
    score = silhouette_score(X_train_2d, y_pred, metric=distance_flag)
    summary_sil_scores.append(score)
    if yi == 0:
        plt.title("silhouette score: " + str(score)[:5], loc='left', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig(outfile + "/" + main_info + f"_k={n_clstr}" + ".jpg")
    plt.close()  # free up the memory

    # Plot the dendrogram with the cut-off for n_clstr clusters
    plt.figure(figsize=(10, 7), dpi=300)
    plt.title(f'Dendrogram for {n_clstr} clusters', fontsize=20)
    dendrogram(linked,
               truncate_mode='lastp',  # show only the last p merged clusters
               p=n_clstr,  # show only the last n_clstr clusters
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               leaf_font_size=18,
               show_contracted=True)
    plt.yticks(fontsize=15)
    plt.savefig(f"{outfile}/{main_info}_dendrogram_k={n_clstr}.jpg")
    plt.close()

    # count and save each label
    print("y_pre", y_pred)
    y_pred += 1
    with open(outfile + "/" + main_info + f"_k={n_clstr}_label_counts.txt", 'w') as f:
        counts = Counter(y_pred)
        for item, count in counts.items():
            f.write(f"{item}: {count}\n")

    # save the label index
    filename = outfile + "/" + main_info + f"_k={n_clstr}_cluster_labels.txt"

    with open(filename, 'w') as fd:
        for i in y_pred:
            fd.write(str(i) + "\n")

# Apply this function to the y-axis using FuncFormatter
formatter = FuncFormatter(three_decimal)
plt.figure(dpi=300)
plt.plot(k, summary_sil_scores, marker='o')


# summary
for i in range(len(k)):
    plt.annotate(round(summary_sil_scores[i], 3),
                 (k[i], summary_sil_scores[i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 bbox=dict(boxstyle='round,pad=0.2',
                           edgecolor='grey', facecolor='white',
                           alpha=0.8))

plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.gca().yaxis.set_major_formatter(formatter)  # set forma
plt.title('Silhouette scores')
plt.grid(True)
print("last stage reached")
plt.savefig(outfile + "/" + main_info + "silhouette_summary.jpg")
plt.show()

