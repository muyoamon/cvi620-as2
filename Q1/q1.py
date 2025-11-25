import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from keras import layers, Sequential

def load_data(path: str, flatten = True):
    data_list = []
    labels = []

    for i, address in enumerate(glob.glob(path)):
        img = cv2.imread(address)
        img = cv2.resize(img, (256,256))
        img = img/255
        img = [img.flatten() if flatten else img][0]

        data_list.append(img)
        label = address.split("/")[-1].split(".")[0]
        labels.append(label)

    data_list = np.array(data_list);

    return data_list, labels


def load_all_data(flatten = True):
    train_X, train_Y = load_data("Q1/train/Cat/*", flatten)
    train_X2, train_Y2 = load_data("Q1/train/Dog/*", flatten)
    test_X1, test_Y1 = load_data("Q1/test/Cat/*", flatten)
    test_X2, test_Y2 = load_data("Q1/test/Dog/*", flatten)
    return np.concatenate((train_X, train_X2)), np.concatenate((test_X1, test_X2)), train_Y + train_Y2, test_Y1 + test_Y2

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
        
        layers.Dense(2, activation='softmax'),
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