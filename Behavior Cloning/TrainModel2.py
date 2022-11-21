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
    raw_labels_x = []
    raw_labels_y = []
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
# print(type_labels)
model.fit(data, type_labels, epochs=5)
pred_label = model.predict(data)
pred_label = np.array([np.argmax(item) for item in pred_label])


# pred_mat = np.zeros
pred_data = data.copy()
pred_data = pred_data.tolist()
for i in range(len(pred_data)):
    temp = np.zeros((10,15))
    temp += pred_label[i]
    pred_data[i].append(temp.tolist())

pred_data = np.array(pred_data, dtype=int)


model1 = keras.Sequential([
    keras.layers.Input(shape=(3,10,15)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(27, activation='softmax'),
])




optimizer = keras.optimizers.Adam(lr=0.0001)
model1.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])



history = model1.fit(pred_data, id_labels, epochs=250)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()

# pred_id = model1.predict(pred_data)


# pred_id = np.array([np.argmax(item) for item in pred_id])
#
# id_labels = np.array([np.argmax(item) for item in id_labels])



model3 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model3.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model3.fit(data, x_labels, epochs=20)


model4 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(15, activation='softmax'),
])

model4.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model4.fit(data, y_labels, epochs=5)


test = list(np.load('new_data.npy', allow_pickle=True))
t_data, t_type_labels, t_id_labels, t_x_labels, t_y_labels = prepare_data(test)

t_pred_label = model.predict(t_data)
t_pred_label = np.array([np.argmax(item) for item in t_pred_label])
t_pred_data = t_data.copy()
t_pred_data = t_pred_data.tolist()
for i in range(len(t_pred_data)):
    temp = np.zeros((10,15))
    temp += t_pred_label[i]
    t_pred_data[i].append(temp.tolist())


t_pred_data = np.array(t_pred_data, dtype=int)

t_pred_id = model1.predict(t_pred_data)
t_pred_id = np.array([np.argmax(item) for item in t_pred_id])

t_pred_x = model3.predict(t_data)
t_pred_x = np.array([np.argmax(item) for item in t_pred_x])

t_pred_y = model4.predict(t_data)
t_pred_y = np.array([np.argmax(item) for item in t_pred_y])


t_type_labels = np.array([np.argmax(item) for item in t_type_labels])
print("accuracy for type ",np.mean(t_type_labels == t_pred_label))

t_id_labels = np.array([np.argmax(item) for item in t_id_labels])
print("accuracy for id ",np.mean(t_id_labels == t_pred_id))

t_x_labels = np.array([np.argmax(item) for item in t_x_labels])
print("accuracy for x ",np.mean(t_x_labels == t_pred_x))

t_y_labels = np.array([np.argmax(item) for item in t_y_labels])
print("accuracy for y ",np.mean(t_y_labels == t_pred_y))
