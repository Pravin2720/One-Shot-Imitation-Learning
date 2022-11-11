import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

block_type = 1
tower = 0
square = 1

def prepare_data(data):
    blocks = []
    raw_labels_type = []
    raw_labels_id = []
    labels_x = []
    labels_y = []
    for seq in data:
        for idx in range(len(seq)):
                block_type = np.array(seq[idx][0])
                block_id = np.array(seq[idx][1])
                # stacked = np.vstack([block_type, block_id])
                # stacked.reshape(20,15,1)
                # blocks.append(stacked)
                blocks.append(np.array([block_type, block_id]))

                raw_labels_type.append(seq[idx][2])
                raw_labels_id.append(seq[idx][3])
                labels_x.append(seq[idx][4])
                labels_y.append(seq[idx][5])
    raw_labels_type = np.array(raw_labels_type, dtype=int)
    labels_type = np.zeros((raw_labels_type.size, 4))
    labels_type[np.arange(raw_labels_type.size), raw_labels_type] = 1

    raw_labels_id = np.array(raw_labels_id, dtype=int)
    labels_id = np.zeros((raw_labels_id.size, 27))
    labels_id[np.arange(raw_labels_id.size), raw_labels_id] = 1
    return np.array(blocks), labels_type, labels_id, np.array(labels_x), np.array(labels_y)

model = keras.Sequential([
    # keras.layers.Input(shape=(20,15,1)),
    # keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax'),
])


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

train = list(np.load('new_data.npy', allow_pickle=True))

data, type_labels, id_labels, x_labels, y_labels = prepare_data(train)
print(data.shape,data[0])
# print(type_labels)
model.fit(data, type_labels, epochs=5)
pred_label = model.predict(data)
pred_label = np.array([np.argmax(item) for item in pred_label])


# pred_mat = np.zeros
pred_data = data.copy()
pred_data = pred_data.tolist()
for i in range(638):
    temp = np.zeros((10,15))
    temp += pred_label[i]
    pred_data[i].append(temp.tolist())

pred_data = np.array(pred_data, dtype=int)


print(pred_data[0])


model1 = keras.Sequential([
    keras.layers.Input(shape=(3,10,15)),
    # keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(27, activation='softmax'),
])



optimizer = keras.optimizers.Adam(lr=0.001)
model1.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])



history = model1.fit(pred_data, id_labels, epochs=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()

pred_id = model1.predict(pred_data)

pred_id = np.array([np.argmax(item) for item in pred_id])

id_labels = np.array([np.argmax(item) for item in id_labels])
for i in range(len(pred_id)):
    print(pred_id[i],id_labels[i])

exit()

# =======================================================================================================================================================
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# =======================================================================================================================================================
#
#
#
#
#
#










pred_label = pred_label.reshape(-1, 1)
t = data.reshape(638,-1)
print(t.shape)
print(pred_label.shape)
i = np.hstack([t,pred_label])
print(i.shape)




model1 = keras.Sequential([
    # keras.layers.Input(shape=(20,15,1)),
    # keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Flatten(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(27, activation='softmax'),
])


optimizer = keras.optimizers.Adam(lr=0.005)
model1.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])



history = model1.fit(i, id_labels, epochs=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()

pred_id = model1.predict(i)
pred_id = np.array([np.argmax(item) for item in pred_id])

id_labels = np.array([np.argmax(item) for item in id_labels])
for i in range(len(pred_id)):
    print(pred_id[i],id_labels[i])
# print(pred_id)
# np.append(data, pred_label, axis=0)


# pred_label = np.reshape(pred_label, (-1,1))
# new_data = np.hstack([data, pred_label])

# print(np.shape(data), np.shape(pred_label))

