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

        # Resize image to 256x256
        img_resized = cv2.resize(img_data, (256, 256))

        data_list.append(img_resized)
        labels.append(label)

    
    

    return data_list, labels


def load_all_data():
    train_data, train_labels = load_data('Q2/mnist_train.csv')
    test_data, test_labels = load_data('Q2/mnist_test.csv')

    x_train = np.array(train_data)
    x_test = np.array(test_data)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    return x_train, x_test, y_train, y_test

def get_cnn_model(input_shape=(256, 256, 3)):
    model = Sequential([
        layers.Conv2D(4, (3, 3), activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.25),


        layers.Conv2D(8, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.25),


        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = get_cnn_model(input_shape=(256,256, 3))
model.summary()



# -- KNN --
# DATA
x_train, x_test, y_train, y_test = load_all_data()

# MODEL
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
knn_predictions = clf.predict(x_test)

#EVAL
knn_accuracy = accuracy_score(y_test, knn_predictions)


# -- CNN --
#DATA
x_train, x_test, y_train, y_test = load_all_data(flatten=False);
le = LabelEncoder();

y_test = le.fit_transform(y_test)
y_train = le.fit_transform(y_train)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)




# MODEL
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=16)




# EVAL
plt.figure(figsize =(10,5))
plt.plot(H.history['accuracy'], label='Training Accuracy')
plt.plot(H.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('CNN_res.png')

print(f" KNN accuracy: {knn_accuracy * 100}%")
print(f" CNN accuracy: {H.history['val_accuracy'][-1] * 100}%")