import pickle
from scipy.ndimage import convolve1d, gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt


with open('mw_rewards_unsmoothed.pkl', 'rb') as f:
    data = pickle.load(f)

for i in data:
    count = 0
    skip = False
    if len(data[i]) == 0:
        continue
    while data[i][count][-1] == 0:
        count += 1
        if count == 50:
            skip = True
            break
    if skip:
        print(f'Check env {i}')
        continue
    
    delta = 4

    filter = (1.0 / delta) * np.array([1] * delta)
    res1 = convolve1d(np.array(data[i][count][0]), filter)

    filter = (1.0 / delta) * np.array([1] * delta + [0] * (delta - 1))
    res2 = convolve1d(np.array(data[i][count][0]), filter)

    filter = (1.0 / delta) * np.array([0] * (delta - 1) + [1] * delta)
    res3 = convolve1d(np.array(data[i][count][0]), filter)

    sigma = 3
    res4 = gaussian_filter1d(np.array(data[i][count][0]), sigma)

    sigma = 6
    res5 = gaussian_filter1d(np.array(data[i][count][0]), sigma)

    sigma = 9
    res6 = gaussian_filter1d(np.array(data[i][count][0]), sigma)

    sigma = 12
    res7 = gaussian_filter1d(np.array(data[i][count][0]), sigma)


    smoothed1 = np.zeros_like(np.array(data[i][count][0]))
    smoothed1[0] = data[i][count][0][0]
    original_max = np.max(np.array(data[i][count][0]))

    alpha = 0.8

    for x in range(1, len(data[i][count][0])):
        smoothed1[x] = alpha * data[i][count][0][x] + (1 - alpha) * smoothed1[x-1]

    smoothed2 = np.zeros_like(np.array(data[i][count][0]))
    smoothed2[0] = data[i][count][0][0]
    original_max = np.max(np.array(data[i][count][0]))

    alpha = 0.6

    for x in range(1, len(data[i][count][0])):
        smoothed2[x] = alpha * data[i][count][0][x] + (1 - alpha) * smoothed2[x-1]


    names = ['original', 'conv_uniform', 'conv_uniform_before', 'conv_uniform_after', 'gauss_sig_3', 'gauss_sig_6', 'gauss_sig_9', 'gauss_sig_12', 'ema_0.8', 'ema_0.6']

    x = np.array([a for a in range(len(res1))])

    for idx, r in enumerate([np.array(data[i][count][0]), res1, res2, res3, res4, res5, res6, res7, smoothed1, smoothed2]):
        plt.plot(x, r, label=names[idx])

    plt.ylabel(f'Rewards for env {i}')
    plt.legend()
    plt.savefig(f'{i}.png')
    plt.close('all')
