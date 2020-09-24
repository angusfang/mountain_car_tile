import numpy as np
import tile_coding
from matplotlib import pyplot as plt

def linear_regression(x, y):
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
    return beta

# create training data
x1 = np.linspace(-1.2, 0.6, 100)
x2 = np.linspace(-0.07, 0.07, 100)
x1, x2 = np.meshgrid(x1, x2)
y = np.cos(x1) + np.cos(x2)
x1 = np.reshape(x1, [-1])
x2 = np.reshape(x2, [-1])
y = np.reshape(y, [-1])

# use tile function
tile_d = tile_coding.Tile1D(number=10)
tile_v = tile_coding.Tile1D(number=10)

while True:
    tile_index_d = tile_d.x(x1)[1]
    tile_index_v = tile_v.x(x2)[1]
    for tile_index_count_d in range(tile_d.number):
        for tile_index_count_v in range(tile_v.number):
            d_x_sort = x1[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
            v_x_sort = x2[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
            y_sort = y[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
            local_input = np.vstack([d_x_sort, v_x_sort])
            local_input = local_input.transpose()

            cal_beta = linear_regression(local_input, y_sort)
            predict = cal_beta[0] + cal_beta[1]*d_x_sort + cal_beta[2]*v_x_sort

            if 'fig' not in locals().keys():
                fig = plt.figure()
            axis = fig.gca(projection='3d')
            axis.scatter3D(d_x_sort, v_x_sort, predict, c=predict, cmap='Greys')
    plt.pause(0.1)
    fig.clf()


