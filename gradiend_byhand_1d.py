import tile_coding
import numpy as np
from matplotlib import pyplot as plt

def partial_w(y_predict, y_true, x_input):
    return 2.0*(y_predict - y_true)* x_input

def partial_b(y_predict, y_true):
    return 2.0 * (y_predict - y_true)

# construct data
x = np.linspace(-1.0, 1.0, 2000)
y = 10*x**2 * np.sin(3*x)+np.random.normal(0, 0.1, x.shape)

# use tile_coding
tile_num = 10
tile = tile_coding.Tile1D(number=tile_num)
weights = np.zeros([tile_num, 2])
# construct model

while True:
    alpha = 0.001
    tile_index = tile.x(x)[1]
    for tile_index_count in range(tile.number):
        x_sort = x[np.where(tile_index == tile_index_count)]
        y_sort = y[np.where(tile_index == tile_index_count)]
        line = weights[tile_index_count]
        y_predict = line[0] + line[1] * x_sort
        for i in range(len(y_sort)):
            line[0] = line[0] - alpha * partial_b(y_predict[i], y_sort[i])
            line[1] = line[1] - 3 * alpha * partial_w(y_predict[i], y_sort[i], x_sort[i])

        y_predict = line[0] + line[1] * x_sort

        plt.scatter(x_sort, y_predict, c='b')
        plt.scatter(x_sort, y_sort, c='r', s=0.1)
    plt.draw()
    plt.pause(0.001)
    plt.clf()




