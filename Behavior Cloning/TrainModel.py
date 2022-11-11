import numpy as np
import tensorflow as tf
from tensorflow import keras


block_type = 1
tower = 0
square = 1

def prepare_data(data):
    blocks = []
    raw_labels_type = []
    labels_id = []
    labels_x = []
    labels_y = []
    for seq in data:
        for idx in range(len(seq)):
                block_type = np.array(seq[idx][0])
                block_id = np.array(seq[idx][1])
                blocks.append(np.array([block_type, block_id]))
                raw_labels_type.append(seq[idx][2])
                labels_id.append(seq[idx][3])
                labels_x.append(seq[idx][4])
                labels_y.append(seq[idx][5])
    labels_type = np.array(raw_labels_type, dtype=int)
    # raw_labels_type = np.array(raw_labels_type, dtype=int)
    # labels_type = np.zeros((raw_labels_type.size, 4))
    # labels_type[np.arange(raw_labels_type.size), raw_labels_type] = 1
    return np.array(blocks), labels_type, np.array(labels_id, dtype=int), np.array(labels_x), np.array(labels_y)

m = tf.keras.Sequential()
m.add(keras.layers.Flatten(input_shape=(2,10,15), name="input"))
m.add(keras.layers.Dense(256,activation='relu', name="layer1"))
m.add(keras.layers.Dense(128, activation='relu', name="layer2"))
m.add(keras.layers.Dense(5, activation='softmax', name="output"))




m1 = tf.keras.Sequential()
m1.add(keras.layers.Flatten(input_shape=(2,10,15), name="input"))
m1.add(keras.layers.Dense(256,activation='relu', name="layer1"))
m1.add(keras.layers.Dense(128, activation='relu', name="layer2"))
m1.add(keras.layers.Dense(5, activation='softmax', name="output"))


model_base = tf.keras.Model(m.output, m1)



# out = m.get_layer('out').output

# model_base = tf.keras.Model(m.input, out)
#
# base_out = model_base(pre1)
# model = tf.keras.Model(input1, base_out)


# def getModel(name):
#     model = keras.Sequential(name=name)
#     model.add(keras.layers.Flatten(input_shape=(2,10,15)))
#     model.add(keras.layers.Dense(256,activation='relu'))
#     model.add(keras.layers.Dense(128, activation='relu'))
#     model.add(keras.layers.Dense(5, activation='softmax'))
#     return model


# model1 = getModel('block-type')

train = list(np.load('new_data.npy', allow_pickle=True))

data, type_labels, id_labels, x_labels, y_labels = prepare_data(train)
print(data.shape)
print(type_labels.shape)
model1 = tf.keras.Sequential()
model1.add(keras.layers.Flatten(input_shape=(2,10,15)))
model1.add(keras.layers.Dense(256,activation='relu'))
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dense(5, activation='softmax'))
# x = tf.keras.layers.Flatten(input_shape=(2,10,15))(data)
# x = tf.keras.layers.Dense(256,activation='relu')(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# out = tf.keras.layers.Dense(5, activation='softmax')(x)
    # return model
model1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model1.fit(data, type_labels, epochs=5)
out = model1.predict(data)
o = tf.argmax(out, axis=1, output_type=tf.dtypes.int32)
print(o)
# exit()
# n = tf.convert_to_tensor(o, dtype=tf.int32)
o = tf.reshape(o,(638,-1))
# print(o)
t = tf.keras.layers.Flatten(input_shape=(2, 10, 15))(data)
# print(t)
l = tf.concat((t,o), axis=1)
print(l)

print(id_labels.shape)
model2 = keras.Sequential()
# model2.add(keras.layers.Flatten(input_shape=(2, 10, 15)))
model2.add(keras.layers.Dense(360, activation='relu'))
model2.add(keras.layers.Dense(180, activation='relu'))
model2.add(keras.layers.Dense(30, activation='softmax'))
optimizer = keras.optimizers.Adam(lr=0.000005)
model2.compile( optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
# model2.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model2.fit(l, id_labels, epochs=100)






# def some_operations(model_1_prediction):
#     # preform your operations
#     # assuming your operations result in a tensor
#     # which has shape matching with model_2's input
#     tensor = model_1_prediction
#     return tensor

# model2 = getModel('block-id')
#
# x = some_operations(model1.output)
# out = model2(x)
# model_1_2 = tf.keras.Model(model1.input, out, name='model-1+2')



# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(2,10,15)),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(4, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
# sparse categorical for 1 2 3 4...

