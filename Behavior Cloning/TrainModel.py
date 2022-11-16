import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def prepare_data(data):
    blocks = []
    raw_labels_type = []
    raw_labels_id = []
    raw_labels_x = []
    raw_labels_y = []
    for seq in data:
        for idx in range(len(seq)):
                block_type = np.array(seq[idx][0])
                block_id = np.array(seq[idx][1])
                info = np.array(seq[idx][6])
                blocks_temp = np.append(block_type.flatten(), block_id.flatten())
                blocks_temp = np.append(blocks_temp, info.flatten())
                blocks.append(np.array(blocks_temp))

                raw_labels_type.append(seq[idx][2])
                raw_labels_id.append(seq[idx][3])
                raw_labels_x.append(seq[idx][4])
                raw_labels_y.append(seq[idx][5])
    raw_labels_type = np.array(raw_labels_type, dtype=int)
    labels_type = np.zeros((raw_labels_type.size, 4))
    labels_type[np.arange(raw_labels_type.size), raw_labels_type] = 1

    raw_labels_id = np.array(raw_labels_id, dtype=int)
    labels_id = np.zeros((raw_labels_id.size, 27))
    labels_id[np.arange(raw_labels_id.size), raw_labels_id] = 1

    raw_labels_x = np.array(raw_labels_x, dtype=int)
    labels_x = np.zeros((raw_labels_x.size, 10))
    labels_x[np.arange(raw_labels_x.size), raw_labels_x] = 1

    raw_labels_y = np.array(raw_labels_y, dtype=int)
    labels_y = np.zeros((raw_labels_y.size, 15))
    labels_y[np.arange(raw_labels_y.size), raw_labels_y] = 1
    return np.array(blocks), labels_type, labels_id, labels_x, labels_y

train = list(np.load('train_data.npy', allow_pickle=True))
data, type_labels, id_labels, x_labels, y_labels = prepare_data(train)


# Training model
type_model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax'),
])
type_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
type_model.fit(data, type_labels, epochs=2)

id_model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(27, activation='softmax'),
])
id_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
id_model.fit(data, id_labels, epochs=5)

x_model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
x_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
x_model.fit(data, x_labels, epochs=3)

y_model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(15, activation='softmax'),
])
y_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
y_model.fit(data, y_labels, epochs=2)


# Testing model
# pred_type = type_model.predict(data)
# pred_type = np.array([np.argmax(item) for item in pred_type])
#
#
# pred_id = id_model.predict(data)
# pred_id = np.array([np.argmax(item) for item in pred_id])
#
# pred_x = x_model.predict(data)
# pred_x = np.array([np.argmax(item) for item in pred_x])
#
# pred_y = y_model.predict(data)
# pred_y = np.array([np.argmax(item) for item in pred_y])
#
#
# labels_type = np.array([np.argmax(item) for item in type_labels])
# print(np.mean(labels_type == pred_type))
#
# labels_id = np.array([np.argmax(item) for item in id_labels])
# print(np.mean(labels_id == pred_id))
#
# labels_x = np.array([np.argmax(item) for item in x_labels])
# print(np.mean(labels_x == pred_x))
#
# labels_y = np.array([np.argmax(item) for item in y_labels])
# print(np.mean(labels_y == pred_y))

# type_model.save('type_model')
# id_model.save('id_model')
# x_model.save('x_model')
# y_model.save('y_model')
