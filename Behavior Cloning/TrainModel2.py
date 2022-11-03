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
    raw_labels_type = np.array(raw_labels_type, dtype=int)
    labels_type = np.zeros((raw_labels_type.size, 4))
    labels_type[np.arange(raw_labels_type.size), raw_labels_type] = 1
    return np.array(blocks), labels_type, np.array(labels_id), np.array(labels_x), np.array(labels_y)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,10,15)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax'),
])


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

train = list(np.load('new_data.npy', allow_pickle=True))

data, type_labels, id_labels, x_labels, y_labels = prepare_data(train)
print(data.shape)
print(type_labels.shape)
model.fit(data, type_labels, epochs=5)
