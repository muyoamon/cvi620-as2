import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from keras import layers, Sequential
import pandas as pd

def load_data(path: str):
    data_list = []
    labels = []

    # Load mnist csv data
    csv = pd.read_csv(path, header=None)
    for index, row in csv.iterrows():
        label = row[0]
        img_data = row[1:].values.astype(np.uint8).reshape(28, 28)


        data_list.append(img_data)
        labels.append(label)

    return data_list, labels


def load_all_data():
    train_data, train_labels = load_data('Q2/mnist_train.csv')
    test_data, test_labels = load_data('Q2/mnist_test.csv')

    x_train = np.array(train_data).astype("float32") / 255
    x_test = np.array(test_data).astype("float32") / 255
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    return x_train, x_test, y_train, y_test

def get_cnn_model(input_shape=(28, 28, 1)):
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPool2D((2,2)),


        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D((2,2)),


        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = get_cnn_model(input_shape=(28,28, 1))
model.summary()


# -- CNN --
#DATA
x_train, x_test, y_train, y_test = load_all_data();
le = LabelEncoder();

y_test = le.fit_transform(y_test)
y_train = le.fit_transform(y_train)

y_test = to_categorical(y_test, num_classes=10)
y_train = to_categorical(y_train, num_classes=10)




# MODEL
H = model.fit(x_train, y_train, epochs=5, batch_size=8, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0)




# EVAL
plt.figure(figsize =(10,5))
plt.plot(H.history['accuracy'], label='Training Accuracy')
plt.plot(H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Q2 CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('q2_CNN_res.png')

print(f" CNN accuracy: {test_acc * 100}%")
print(f" CNN loss: {test_loss * 100}%")