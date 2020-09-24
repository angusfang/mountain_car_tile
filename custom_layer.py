
my_inputs = []
lines = []
local_models = []
# construct inputs
for i in range(tile.number):
    my_inputs.append(Input(shape=[1]))  # for weight
    my_inputs.append(Input(shape=[1]))  # for bia
# construct model
for i in range(tile.number):
    weight = layers.Dense(1, use_bias=False)(my_inputs[i])
    bia = layers.Dense(1, use_bias=False)(my_inputs[i + tile.number])
    line = layers.Add()([weight, bia])
    local_model = Model(inputs=my_inputs, outputs=line)
    local_model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.1), loss=tf.losses.mean_squared_error)
    local_models.append(local_model)

    lines.append(layers.Dense(1, use_bias=False)(line))

o = layers.Add()(lines)
global_model = Model(inputs=my_inputs, outputs=o)
global_model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.1), loss=tf.losses.mean_squared_error)
# ia = Input(shape=[None, 1])
# ib = Input(shape=[None, 1])
# a = layers.Dense(1)(ia)
# b = layers.Dense(1)(ib)
# o = layers.Add()([a, b])

# test