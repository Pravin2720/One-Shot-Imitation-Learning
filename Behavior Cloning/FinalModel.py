import numpy as np
# import tensorflow as tf
# from tensorflow import keras

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
    labels_type = np.zeros((raw_labels_type.size, 4), dtype=int)
    labels_type[np.arange(raw_labels_type.size), raw_labels_type] = 1

    raw_labels_id = np.array(raw_labels_id, dtype=int)
    labels_id = np.zeros((raw_labels_id.size, 27), dtype=int)
    labels_id[np.arange(raw_labels_id.size), raw_labels_id] = 1

    raw_labels_x = np.array(raw_labels_x, dtype=int)
    labels_x = np.zeros((raw_labels_x.size, 10), dtype=int)
    labels_x[np.arange(raw_labels_x.size), raw_labels_x] = 1

    raw_labels_y = np.array(raw_labels_y, dtype=int)
    labels_y = np.zeros((raw_labels_y.size, 15), dtype=int)
    labels_y[np.arange(raw_labels_y.size), raw_labels_y] = 1

    labels = np.concatenate((labels_type, labels_id, labels_x, labels_y), axis=1)
    return np.array(blocks), labels

train = list(np.load('train_data.npy', allow_pickle=True))
data, labels = prepare_data([train[0]])

#
# model = keras.Sequential([
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(56, activation='softmax'),
# ])
# model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(data, labels, epochs=20)
