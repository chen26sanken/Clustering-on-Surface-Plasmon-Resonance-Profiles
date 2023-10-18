import os
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from tslearn.clustering import TimeSeriesKMeans
from matplotlib.ticker import FuncFormatter


def subtract_40(x, pos):  # the tick formatter function
    return f'{int(0.1*x + 40)}'


def three_decimal(x, pos):  # formats a number to three decimal places
    return f'{x:.3f}'


# infile = "preprocessed_data/0628SPR_fft_coef_5.csv"
infile = "preprocessed_data/0628SPR_fft_rev5.csv"
# infile = "preprocessed_data/0628spikes_cut.csv"
outfile = "entry4_kmeans_fft_rev5"


k = list(range(2, 12))
df = pd.read_csv(infile)
df = df.iloc[:, :]
df = df.T.ffill()  # transpose and fill the nan using next row value
indata = "all"

df2np = df.to_numpy()
df2np_reshape = np.reshape(df2np, (2000, df2np.shape[1], 1))  # (samples, data len, 1)

seed = 1000
X_train = df2np_reshape

print("shape of training data", X_train.shape)

# deal with complex number
X_train_complex = np.vectorize(np.complex)(X_train)
X_train_real = np.concatenate((np.real(X_train_complex),
                               np.imag(X_train_complex)), axis=1)  # Concatenate the real and imaginary parts
X_train = X_train_real

X_train = TimeSeriesScalerMeanVariance(mu=0, std=1).fit_transform(X_train)
X_train = TimeSeriesResampler(sz=400).fit_transform(X_train)
print("After shape of training data", X_train.shape)


X_train_reshaped = X_train.reshape(2000, 400)
X_train_save = pd.DataFrame(X_train_reshaped)
X_train_save.to_csv('check_input.csv', index=False)  # check input values

os.makedirs(outfile, exist_ok=True)
summary_sil_scores = []
figs_per_row = 3

for n_clstr in k:
    print(f"K-Means Clustering for {n_clstr} clusters")
    kmeans = TimeSeriesKMeans(n_clusters=n_clstr, verbose=True, random_state=seed)
    y_pred = kmeans.fit_predict(X_train)

    n_rows = (n_clstr - 1) // figs_per_row + 1
    fig_height = 6 * n_rows
    plt.figure(dpi=500, figsize=(24, fig_height))

    counts = Counter(y_pred)  # count labels

    for yi in range(n_clstr):
        ax = plt.subplot(n_rows, figs_per_row, yi + 1)
        cluster_data = X_train[y_pred == yi]
        for xx in cluster_data:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(kmeans.cluster_centers_[yi].ravel(), "r-")

        # Add cluster count and label to plot
        plt.text(0.5, 0.95, f'Cluster {yi + 1} (Count: {counts[yi]})',
                 transform=plt.gca().transAxes, fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.tight_layout()

    # Calculate silhouette score and append to list
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    score = silhouette_score(X_train_2d, y_pred)
    summary_sil_scores.append(score)
    if yi == 0:
        plt.title("silhouette score: " + str(score)[:5], loc='left', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig(outfile + f"/kmeans_k={n_clstr}" + ".jpg")
    plt.close()  # free up the memory

    print("y_pre", y_pred)
    y_pred += 1
    with open(outfile + f"/kmeans_k={n_clstr}_label_counts.txt", 'w') as f:
        counts = Counter(y_pred)
        for item, count in counts.items():
            f.write(f"{item}: {count}\n")  # count and save each label

    filename = outfile + f"/kmeans_k={n_clstr}_cluster_labels.txt"
    with open(filename, 'w') as fd:
        for i in y_pred:
            fd.write(str(i) + "\n")  # save the label index


formatter = FuncFormatter(three_decimal)
plt.figure(dpi=300)
plt.plot(k, summary_sil_scores, marker='o')  # plot the summary scores

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
plt.savefig(outfile + "/kmeans_silhouette_summary.jpg")
plt.show()
