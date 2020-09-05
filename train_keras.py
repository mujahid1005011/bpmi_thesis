import numpy as np
import tensorflow as tf
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder

# Importing the dataset
# dataset = np.genfromtxt("housing.csv", delimiter=None)

dataset = pd.read_excel('BPMI_Thesis_train_data.xlsx').to_numpy()
X = np.concatenate([dataset[:, 0:3], dataset[:, 4:5]], axis=-1)
y = dataset[:, 5]

y = y.astype(np.float)

# Splitting the dataset into the Training set and Test set
X_train = X
y_train = y

test_dataset = pd.read_excel('BPMI_Thesis_test_data.xlsx').to_numpy()
X_test = np.concatenate([test_dataset[:, 0:3], test_dataset[:, 4:5]], axis=-1)
y_test = test_dataset[:, 5]

# print(y)


try:
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation='relu', input_dim=4))

    # Adding the second hidden layer
    model.add(Dense(units=32, activation='relu'))

    # Adding the third hidden layer
    model.add(Dense(units=32, activation='relu'))

    # Adding the output layer

    model.add(Dense(units=1))

    # model.add(Dense(1))
    # Compiling the ANN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the ANN to the Training set
    # print(X_train)
    model.fit(X_train, y_train, batch_size=10, epochs=100)

    y_pred = model.predict(X_test)
    # print(y_pred)
    slots = {
        100: 0,
        90: 0,
        80: 0,
        70: 0,
        60: 0,
        50: 0,
        40: 0,
        30: 0,
        20: 0,
        10: 0
    }
    index = 0
    import math

    for y in y_test:
        y_p = y_pred[index][0]
        diff = 1 if y == 0 else (abs(y - y_p) * 100 / y)
        index += 1
        included = False
        for x in slots.keys():
            if diff >= x:
                slots[x] += 1
                included = True
                break
        if not included:
            slots[10] += 1

    import copy

    tests_len = len(y_test)
    slots_per = copy.deepcopy(slots)
    for x in slots.keys():
        slots_per[x] = (slots[x] * 100.0) / (tests_len * 1.0)

    print(slots_per)
    print(slots)

    plt.plot(y_test, color='red', label='Actual data')
    plt.plot(y_pred, color='blue', label='Predicted data')
    plt.title('Prediction')
    plt.legend()
    plt.show()



except Exception as e:
    import traceback

    print(traceback.format_exc())
