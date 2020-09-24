import numpy as np
import tile_coding
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from matplotlib import pyplot as plt

# create training data
x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)
x1, x2 = np.meshgrid(x1, x2)
y = np.cos(x1) + np.cos(x2)
x1 = np.reshape(x1, [-1])
x2 = np.reshape(x2, [-1])
y = np.reshape(y, [-1])

# use tile function
tile_d = tile_coding.Tile1D(number=10)
tile_v = tile_coding.Tile1D(number=10)

# construct model
planes = []
for i in range(tile_d.number):
    planes.append([])
    for j in range(tile_v.number):
        input_x = Input(shape=[2])
        o = layers.Dense(1)(input_x)
        plane = Model(input_x, o)
        plane.compile(optimizer=tf.optimizers.SGD(learning_rate=0.05, momentum=0.5), loss=tf.losses.mean_squared_error)
        planes[i].append(plane)

while True:
    tile_index_d = tile_d.x(x1)[1]
    tile_index_v = tile_v.x(x2)[1]
    for tile_index_count_d in range(tile_d.number):
        for tile_index_count_v in range(tile_v.number):
            d_x_sort = x1[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
            v_x_sort = x2[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
            y_sort = y[np.where((tile_index_d == tile_index_count_d) & (tile_index_v == tile_index_count_v))]
            local_model = planes[tile_index_count_d][tile_index_count_v]
            local_input = np.vstack([d_x_sort, v_x_sort])
            local_input = local_input.transpose()
            local_model.fit(local_input, y_sort)

            predict = local_model(local_input)
            if 'fig' not in locals().keys():
                fig = plt.figure()
            axis = fig.gca(projection='3d')
            axis.scatter3D(d_x_sort, v_x_sort, predict, c=predict, cmap='Greys')
    plt.pause(0.1)
    fig.clf()


