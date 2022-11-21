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
                
    labels = np.vstack((raw_labels_type, raw_labels_id, raw_labels_x, raw_labels_y)).transpose()
    return np.array(blocks), labels

train = list(np.load('square_train_data.npy', allow_pickle=True))
data,labels = prepare_data(train)

model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu'), 
    keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(4, activation='relu'),
])


optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['accuracy'])

history = model.fit(data, labels, epochs=5)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()


model.save('square_final_model')
