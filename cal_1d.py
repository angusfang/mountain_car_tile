import tile_coding
import numpy as np
from matplotlib import pyplot as plt


def linear_regression(x, y):
    x = np.concatenate((np.ones((x.shape[0], 1)), x[:, np.newaxis]), axis=1)
    y = y[:, np.newaxis]
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
    return beta


# construct data
x = np.linspace(-1.0, 1.0, 2000)
y = 10 * x ** 2 * np.sin(3 * x) + np.random.normal(0, 0.1, x.shape)

# use tile_coding
tile = tile_coding.Tile1D(number=10)

while True:

    tile_index = tile.x(x)[1]
    for tile_index_count in range(tile.number):
        x_sort = x[np.where(tile_index == tile_index_count)]
        y_sort = y[np.where(tile_index == tile_index_count)]
        cal_beta = linear_regression(x_sort, y_sort)
        predict = cal_beta[0] + x_sort * cal_beta[1]

        plt.scatter(x_sort, predict, c='b')
        plt.scatter(x_sort, y_sort, c='r', s=0.1)
    plt.draw()
    plt.pause(0.001)
    plt.clf()
