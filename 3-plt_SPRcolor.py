import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # for custom legend
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler


outfile = "0628_SPR_check"
main_info = "entry6_hier_ward_euclid_fft_rev5"
os.makedirs(outfile, exist_ok=True)

# load data
df = pd.read_csv("preprocessed_data/0628spikes_cut.csv")
df = df.T  # drop odd number columns and transpose
dataset = df.ffill(axis=1)  # fill the nan using next column value
dataset = dataset.bfill()

X = dataset.iloc[:,:].values
df_2 = pd.DataFrame(data=X).T  # transpose for the fft transformation

df_label = pd.read_csv(main_info + "/euclidean_k=16_cluster_labels.txt", header=None)
label = df_label.iloc[0:, 0].values  # labels

# create a TimeSeriesScalerMeanVariance object with mu=0 and std=2
scaler = TimeSeriesScalerMeanVariance(mu=0, std=1)
df_scaled = scaler.fit_transform(df_2.T).T.squeeze()  # apply the scaler to each row of the dataframe
df_scaled = pd.DataFrame(data=df_scaled)  # check if correct

plt.figure(figsize=(5,7))

y = df_2.values  # the original
# y = df_scaled.values  # the standardized data
x = range(len(y))

# Define the labels you want to highlight and their corresponding colors
# target_labels = [16]
# target_labels = [8, 11, 15, 16]
# target_labels = [1,2,3,4,5,6,7,9,10,12,13,14]
target_labels = [13,14]
colors = ['orange', 'teal']
# colors = [ 'blue', 'green','red', 'skyblue', 'magenta',
#          'yellow', 'cyan', 'purple', 'brown', 'pink', 'orange', 'teal']

# colors = ['red', 'green', 'blue', 'skyblue', 'magenta',
#          'yellow', 'cyan', 'purple', 'brown', 'pink', 'orange', 'teal']

# Create a color map using a dictionary
color_map = {target_label: color for target_label, color in zip(target_labels, colors)}
default_color = "black"

# Iterate over the SPR profiles and plot them with the appropriate color
for i in range(y.shape[1]):
    plt.plot(x, y[:, i], color=color_map.get(label[i], default_color),
             alpha=0.5 if label[i] in target_labels else 0.02)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# plt.ylim(-3, 3)
plt.ylim(-5, 2)
# plt.ylim(-3, 5)
plt.xlim(100, 400)
plt.xlim(150, 400)

# Create legend outside the plot
legend_elements = [Patch(facecolor=color_map.get(target_label, 'grey'), edgecolor='k',
                         label=f'Label {target_label}') for target_label in target_labels]
plt.legend(handles=legend_elements, loc='upper left')

plt.savefig(outfile + "/" + main_info + ".jpg")
plt.show()
plt.close()
