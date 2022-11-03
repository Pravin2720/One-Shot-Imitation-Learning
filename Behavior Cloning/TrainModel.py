import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train = list(np.load('../new_data.npy', allow_pickle=True))

max_seq_len = 10

def prepare_data(data):
    block_types = []
    block_ids = []
    seq_len = []

    type_i = []
    type_o = []
    for seq in data:
        seq_input_type_grid = []
        input_id_grid = []
        seq_output_type = []
        output_id = []
        output_x = []
        output_y = []
        seq_len.append(len(seq))

        for idx in range(max_seq_len):
            if idx < seq_len[-1]:
                block_type = np.array(seq[idx][0])[:,:10]
                block_id = seq[idx][1][:,:10]
                # print("block_type",block_type)
                # print("block_id",block_id)
                # print("out type",seq[idx][2])
                # print("out id",seq[idx][3])
                # print("out x",seq[idx][4])
                # print("out y",seq[idx][5])
                # converting to one hot

                o_type_labels = np.zeros(4, dtype=int)
                o_type_labels[seq[idx][2]] = 1
                # print(np.hstack([block_type, block_id]).flatten().shape)
                seq_input_type_grid.append(block_type.flatten())
                # seq_input_type_grid.append(np.hstack([block_type, block_id]).flatten())
                seq_output_type.append(o_type_labels)
            else:
                block_type = np.array(seq[seq_len[-1]-1][0])[:, :10]
                seq_input_type_grid.append(block_type.flatten())
                # seq_input_type_grid.append(np.zeros(100))
                seq_output_type.append(np.zeros(4))
        type_i.append(seq_input_type_grid)
        type_o.append(seq_output_type)
    return np.array(type_i), np.array(type_o), seq_len

model = keras.models.Sequential()
def train_model(data,labels):
    model.add(keras.Input(shape=(10, 100)))
    model.add(layers.SimpleRNN(100, return_sequences=True, activation='relu'))
    model.add(layers.SimpleRNN(30, return_sequences=True, activation='relu'))
    model.add(layers.Dense(4))
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(data, labels, epochs=10)
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)


data, labels, seq_len = prepare_data(train)
print(data.shape, labels.shape)
# for s in seq_len:
train_model(data, labels)
