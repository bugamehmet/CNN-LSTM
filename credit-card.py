import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization
import tensorflow as tf

#dataset --> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

veri = pd.read_csv("creditcard.csv")

label_encoder = LabelEncoder().fit(veri.Class)
labels = label_encoder.transform(veri.Class)
classes = list(label_encoder.classes_)

veri = veri.drop(["Class"], axis=1)
nb_features = 30
nb_classes = len(classes)

scaler = StandardScaler().fit(veri.values)
veri = scaler.transform(veri.values)

X_train, X_valid, y_train, y_valid = train_test_split(veri, labels, test_size=0.3)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

print("X_train shape:", X_train.shape)
print("y_valid shape:", X_valid.shape)

X_train = np.array(X_train).reshape(199364, 30, 1)
X_valid = np.array(X_valid).reshape(85443, 30, 1)

model = Sequential()
model.add(Conv1D(512, 1, input_shape=(nb_features, 1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 1))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(LSTM(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.30))
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

score = model.fit(X_train, y_train, epochs = 2, validation_data = (X_valid, y_valid))

print(("Ortalama doğrulama Kaybı: ", np.mean(model.history.history["val_loss"])))
print(("Ortalama doğrulama Başarımı: ", np.mean(model.history.history["val_accuracy"])))


plt.plot(model.history.history["accuracy"], color = "y")
plt.plot(model.history.history["val_accuracy"], color = "b")
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"], color = "g")
plt.plot(model.history.history["val_loss"], color = "r")
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()
