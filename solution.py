# import basic package
import numpy as np
import pandas as pd

# import ML package
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score

# import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam, rmsprop
from keras.layers import Dense, Flatten, Dropout, Concatenate, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Sequential
from keras.applications import VGG16, VGG19, ResNet50, Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers.normalization import BatchNormalization

# Load the data
train_data = pd.read_json("./data/train.json")
test_data = pd.read_json("./data/test.json")

# Generate the training data
train_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_data["band_1"]])
train_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_data["band_2"]])
train_band_3 = (train_band_1 + train_band_2)/2
train_band = np.concatenate([train_band_1[:,:,:, np.newaxis],
                            train_band_2[:,:,:, np.newaxis],
                            train_band_3[:,:,:, np.newaxis]], axis = -1)
train_angle = pd.to_numeric(train_data['inc_angle'], errors='coerce')
train_angle = train_angle.fillna(method='pad')
train_target = np.array(train_data['is_iceberg'])

# Generate the test data
test_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_1"]])
test_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_2"]])
test_band_3 = (test_band_1 + test_band_2)/2
test_band = np.concatenate([test_band_1[:,:,:, np.newaxis],
                           test_band_2[:,:,:, np.newaxis],
                           test_band_3[:,:,:, np.newaxis]], axis = -1)
test_angle = pd.to_numeric(test_data['inc_angle'], errors='coerce')


X_train, X_valid, angle_train, angle_valid, y_train, y_valid = train_test_split(train_band, train_angle, train_target,
    random_state = 1, train_size = 0.8)

#check point
checkpointer = ModelCheckpoint(filepath="saved_model/weights.best.from_scratch.hdf5", verbose=1, save_best_only=True)


def getVGG16concatenateAngleModel():
    input_angle = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_angle)
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=train_band.shape[1:], classes=1)
    x = base_model.get_layer('block5_pool').output
    x = GlobalMaxPooling2D()(x)
    merge = Concatenate()([x, angle_layer])
    merge = Dropout(0.2)(merge)
    pred = Dense(1, activation='sigmoid')(merge)

    opt = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Model(inputs=[base_model.input, input_angle], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def benchmark_model():
    bn_model = 0
    p_activation = "relu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")

    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPooling2D()(img_1)

    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPooling2D()(img_2)

    img_concat = (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))

    dense_ayer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(256, activation=p_activation)(img_concat)))
    dense_ayer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(64, activation=p_activation)(dense_ayer)))
    output = Dense(1, activation="sigmoid")(dense_ayer)

    model = Model([input_1, input_2], output)
    optimizer = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# define model
Model = getVGG16concatenateAngleModel()

Model.fit([X_train, angle_train], y_train,
                batch_size=20,
                epochs=100,
                shuffle=True,
                verbose=1,
                callbacks=[checkpointer])



Model.load_weights("saved_model/weights.best.from_scratch.hdf5")

pred = Model.predict([X_valid, angle_valid])
cls = lambda x: 1 if x >= 0.5 else 0
pred = [cls(value) for value in pred]
accuracy_score(pred, y_valid)