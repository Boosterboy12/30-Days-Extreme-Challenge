# --- IMPORTING OS --- #
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- IMPORTING THE DEPENDENCIES -- #
import tensorflow
import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- IMPORTING THE LAYERS --- #
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# --- IMPORTING THE DATASET --- #
from keras.datasets import mnist

# =========================================================================================================================== #
# =========================================================================================================================== #

# --- INITIALIZE SEQUENTIAL MODEL --- #
model = Sequential()

# --- FIRST FEATURE EXTRACTION LAYER BLOCK --- #
model.add(Conv2D(20, kernel_size=(5, 5), padding="valid", activation="gelu"))
model.add(MaxPooling2D(pool_size=2, padding="valid", strides=2))

# --- SECOND FEATURE EXTRACTION LAYER BLOCK --- #
model.add(Conv2D(60, kernel_size=(5, 5), padding="valid", activation="gelu"))
model.add(MaxPooling2D(pool_size=2, padding="valid", strides=2))

# --- FLATTEN THE MATRICES INTO A 1D VECTOR --- #
model.add(Flatten())

# --- FULLY CONNECTED DENSE LAYERS --- #
model.add(Dense(100, activation="gelu"))
model.add(Dense(50, activation="gelu"))

# --- REGULARIZATION TO IMPROVE GENERALIZATION --- #
model.add(keras.layers.Dropout(0.2))

# --- OUTPUT LAYER WITH SOFTMAX FOR 10 DIGIT CLASSES --- #
model.add(Dense(10, activation="softmax"))

# --- PRINT MODEL TOPOLOGY SUMMARY --- #
model.build(input_shape=(None, 28, 28, 1))
model.summary()

# =========================================================================================================================== #
# =========================================================================================================================== #

# --- LOAD THE HANDWRITTEN DIGITS DATASET --- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# --- NORMALIZE PIXEL VALUES TO RANGE 0-1 --- #
X_train = X_train / 255.0
X_test = X_test / 255.0

# --- RESHAPE ARRAYS TO ADD EXPLICIT CHANNEL DIMENSION --- #
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# =========================================================================================================================== #
# =========================================================================================================================== #

# --- COMPILE MODEL WITH OPTIMIZER AND LOSS FUNCTION --- #
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# --- CALLBACKS FOR BETTER TRAINING --- #
lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=2, verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy", patience=4, restore_best_weights=True
)

# --- TRAIN MODEL --- #
model.fit(
    X_train,
    y_train,
    epochs=15,
    validation_split=0.2,
    callbacks=[lr_scheduler, early_stop],
)

# =========================================================================================================================== #
# =========================================================================================================================== #

# --- EVALUATE MODEL ON UNSEEN TEST DATA --- #
loss, accuracy = model.evaluate(X_test, y_test)

# --- PRINT FINAL PERFORMANCE ACCURACY METRICS --- #
print("Test Accuracy:", accuracy)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --- SAVING THE MODEL --- #
model.save("mnist_cnn.keras")

# =========================================================================================================================== #
#                                                          0 END 1                                                            #
# =========================================================================================================================== #
