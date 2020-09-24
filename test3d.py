import numpy as np
import tile_coding
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from matplotlib import pyplot as plt

# test 3d plot
# fig = plt.figure()
# axis = fig.gca(projection='3d')
x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)
# X1, X2 = np.meshgrid(x1, x2)
# Z = np.cos(X1) + np.cos(X2)
# surface = axis.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='Greys')
# plt.show()

# create training data
s_d = []
s_v = []
s_f = []
for i in x1:
    for j in x2:
        k = np.cos(i) + np.cos(j)
        s_d.append(i)
        s_v.append(j)
        s_f.append(k)
s_d = np.array(s_d)
s_v = np.array(s_v)
s_f = np.array(s_f)

# find approximate function
tile_d = tile_coding.Tile1D(number=50)
tile_v = tile_coding.Tile1D(number=50)
tile_dv = tile_coding.Tile2D(tile_d, tile_v)

my_input = Input(shape=[tile_d.number, tile_v.number])
d = layers.Flatten()(my_input)
d = layers.Dense(200)(d)
d = layers.Dense(100)(d)
o = layers.Dense(1)(d)
my_model = Model(my_input, o)

my_model.compile(optimizer=tf.optimizers.SGD(), loss=tf.losses.mean_squared_error)


while True:
    f = tile_dv.x(s_d, s_v)
    my_model.fit(f, s_f, batch_size=32, epochs=5)
    z = my_model.predict(f)

    if 'fig' not in locals().keys():
        fig = plt.figure()
    axis = fig.gca(projection='3d')
    axis.scatter3D(s_d, s_v, z, c=z, cmap='Greys')
    plt.pause(0.1)
    fig.clf()


