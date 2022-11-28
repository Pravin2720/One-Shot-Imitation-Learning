from tensorflow import keras
import numpy as np


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


model = keras.models.load_model('model')

test = list(np.load('testing_data.npy', allow_pickle=True))
test_tower = test[:16000]
test_square = test[16000:]
data, labels = prepare_data(test_tower)

final_model = keras.models.load_model('final_model')
results = final_model.evaluate(data, labels)
print("Accuracy for making towers",results[1])


# results = model.evaluate(data, labels)
# print("Accuracy for making towers",results[1])

data, labels = prepare_data(test_square)

square_final_model = keras.models.load_model('square_final_model')
results = square_final_model.evaluate(data, labels)
print("Accuracy for making towers",results[1])
# results = model.evaluate(data, labels)
# print("Accuracy for making square",results[1])