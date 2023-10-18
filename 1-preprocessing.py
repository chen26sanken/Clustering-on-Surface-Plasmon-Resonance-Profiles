import os
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler


def filter_signal(x, threshold, to_real=True, fig_index=None):
    n = len(x)
    fft = np.fft.fft(x, n)
    # print(fft, 'fft')
    # os.makedirs(outfile + "/fft_org", exist_ok=True)
    # # plot each figure
    # plt.plot(range(400, 800), fft)  # Plot the data
    # plt.xlabel('Frequency')
    # plt.xticks([])  # Remove xtick labels
    # plt.ylabel('Magnitude')
    # plt.ylim(-500, 500)
    # plt.title(f"molecule_index {fig_index}")
    # plt.savefig(outfile + "/fft_org/"f'fft_{fig_index}.png', dpi=500)
    # # plt.show()  # display the last plot
    # plt.clf()

    # noise reduction
    mask = np.zeros(n)
    mask[:threshold] = 1
    # print(mask, "mask")

    fft *= mask
    # print(PSD, "PSD")
    # os.makedirs(outfile + "/fft_denosiedfft", exist_ok=True)
    # # plot each figure
    # plt.plot(range(400, 800), fft)  # Plot the data
    # plt.xlabel('Frequency')
    # plt.xticks([])  # Remove xtick labels
    # plt.ylabel('Magnitude')
    # plt.ylim(-500, 500)
    # plt.title(f"molecule_index {fig_index} high freq remov")
    # plt.savefig(outfile + "/fft_denosiedfft/"f'fft_{fig_index}.png', dpi=500)
    # # plt.show()  # display the last plot
    # plt.clf()

    # reversed signal
    fft = np.fft.ifft(fft)
    # os.makedirs(outfile + "/fft_reversedfft", exist_ok=True)
    # # plot each figure
    # plt.plot(range(400, 800), fft)  # Plot the data
    # plt.xlabel('Frequency')
    # plt.xticks([])  # Remove xtick labels
    # plt.ylabel('Magnitude')
    # plt.ylim(-500, 500)
    # plt.title(f"molecule_index {fig_index} high freq remov")
    # plt.savefig(outfile + "/fft_reversedfft/"f'fft_{fig_index}.png', dpi=500)
    # # plt.show()  # display the last plot
    # plt.clf()

    if to_real:
        fft = fft.real
    return fft


def filter_signal_coe(x, threshold, to_real=True, fig_index=None):
    n = len(x)
    fft = np.fft.fft(x, n)
    mask = np.zeros(n)  # noise reduction
    mask[:threshold] = 1
    fft *= mask
    fft_coefficient = fft

    return fft_coefficient


# Define the tick formatter function
def axis_adjust(x, pos):
    return f'{int(0.1*x)}'


infile = "data/combined_dataFc2-1.csv"
main_info = "0628"
outfile = "plots_check"

x_lower = 300
x_upper = 1000


# load data
df = pd.read_csv(infile)
df = df.T
dataset = df.ffill(axis=1)  # fill the nan using next column value
dataset = dataset.bfill()

X = dataset.iloc[:, x_lower:x_upper].values  # take the association and dissociation part
print("X", X)
df_fft = pd.DataFrame(data=X).T  # transpose for later fft
print("fft", df_fft)

os.makedirs(outfile, exist_ok=True)
y = df_fft.iloc[:, :].values  # exclude the first index line
x = range(x_lower, x_upper)
fig1, ax = plt.subplots()
ax.plot(x, y)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(axis_adjust))
ax.set_ylim(-200, 1300)
plt.title("original SPR")
# plt.savefig(outfile + "/" + main_info + 'fft01.png', dpi=500)
plt.show()
plt.close()


os.makedirs("preprocessed_data", exist_ok=True)

# remove spikes by cutting specific time window
slice_col0 = y[x_lower-x_lower:490-x_lower, :]
slice_col1 = y[500-x_lower:510-x_lower, :]
slice_col2 = y[550-x_lower:675-x_lower, :]
slice_col3 = y[705-x_lower:800-x_lower, :]
# slice_col4 = y[800-x_lower:800-x_lower, :]
slice_combine = np.vstack((slice_col0, slice_col1, slice_col2, slice_col3))
slice_combine = slice_combine

df = pd.DataFrame(slice_combine)
# df.to_csv("preprocessed_data/0628spikes_cut.csv", index=False)


y = slice_combine
print("slice combine: x shape", y.shape)
x = range(x_lower, x_lower + len(y))  # x is now shifted by x_lower
fig1, ax = plt.subplots()
ax.plot(x, y)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(axis_adjust))
ax.set_ylim(-200, 1300)
plt.title("original SPR cut")
# plt.savefig(outfile + "/" + main_info + 'fft01_cut.png', dpi=500)
plt.show()
plt.close()

# (standardization was removed to the later process, not here)


threshold = 2
slice_combine_df = pd.DataFrame(slice_combine)
df_clean = slice_combine_df.apply(lambda x: filter_signal(x, threshold))  # apply fft_denoiser
y = df_clean.iloc[:, :].values
x = range(x_lower, x_lower + len(y))  # x is now shifted by x_lower
fig3, ax = plt.subplots()
ax.plot(x, y)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(axis_adjust))
ax.set_ylim(-200, 1300)
plt.title(f"Reconstructed SPR (bases {threshold})")
# plt.savefig(outfile + "/" + main_info + 'fft03.png', dpi=500)
plt.show()
plt.close()


# obtain clean SPR data
fft_SPR = pd.DataFrame(df_clean)
print("fft_SPR", fft_SPR.shape)
fft_SPR.to_csv("preprocessed_data/" + main_info + "SPR_fft_rev" + str(threshold) + ".csv", index=False)

# obtain fft coefficient
df_clean_coe = slice_combine_df.apply(lambda x: filter_signal_coe(x, threshold))
y = df_clean_coe.iloc[:, :].values
x = range(len(y))
fig4, ax = plt.subplots()
ax.plot(x, y)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(axis_adjust))
plt.title(f"fft coefficient (bases {threshold})")
# plt.savefig(outfile + "/" + main_info + 'fft04.png', dpi=500)
plt.show()
plt.close()
SPR_fft_coe = pd.DataFrame(df_clean_coe)
# SPR_fft_coe.to_csv("preprocessed_data/" + main_info + "SPR_fft_coef_" + str(threshold) + ".csv", index=False)
