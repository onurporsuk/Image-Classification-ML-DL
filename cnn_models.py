from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.callbacks import CSVLogger
from matplotlib import pyplot as plt
from keras import backend as k
import numpy as np
import main
import cv2
import os


DATADIR = "C:/Users/Onur/PycharmProjects/ImageClassification/dataset"
CATEGORIES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
IMAGE_SIZE = 100

TEST_SIZE = 0.1  # Splitting dataset into train and test with portion
VALIDATION_SIZE = 0.5  # Splitting test data to test and validation with portion

BATCH_SIZE_SMALL_CNN_MODEL = 64
EPOCHS_SMALL_CNN_MODEL = 5

BATCH_SIZE_LARGE_CNN_MODEL = 32
EPOCHS_LARGE_CNN_MODEL = 18

BATCH_SIZE_TL_CNN_MODEL = 8
EPOCHS_TL_CNN_MODEL = 2

K_FOLD_NUMBER = 5
K_FOLD_EPOCHS = 12
K_FOLD_VALIDATION_SIZE = 0.1

EVALUATION_BATCH_SIZE = 8


def create_compile_small_cnn_model(X_train, X_val, y_train, y_val):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(len(CATEGORIES), activation="softmax"))

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    csv_logger = CSVLogger('training_log/log_small_cnn_model.csv', append=False, separator=';')

    print("\nTraining:\n")
    model.fit(X_train, y_train, batch_size=BATCH_SIZE_SMALL_CNN_MODEL, epochs=EPOCHS_SMALL_CNN_MODEL, verbose=1,
              validation_data=(X_val, y_val), callbacks=[csv_logger])

    plot_evaluation(model)

    return model


def create_large_cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(len(CATEGORIES), activation='softmax'))

    model.summary()

    return model


def compile_large_cnn_model(X_train, X_val, y_train, y_val):
    model = create_large_cnn_model()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    csv_logger = CSVLogger('training_log/log_large_cnn_model.csv', append=False, separator=';')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)

    print("\nTraining:\n")
    model.fit(X_train, y_train, batch_size=BATCH_SIZE_LARGE_CNN_MODEL, epochs=EPOCHS_LARGE_CNN_MODEL, verbose=1,
              validation_data=(X_val, y_val), callbacks=[csv_logger])

    plot_evaluation(model)

    return model


def k_fold_validate(X, y):
    skf = StratifiedKFold(n_splits=K_FOLD_NUMBER, random_state=7, shuffle=True)

    fold = 1
    fold_accuracy_values = []
    model = create_large_cnn_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    csv_logger = CSVLogger('training_log/log_large_cnn_model_k_fold.csv', append=False, separator=';')

    for train_index, val_index in skf.split(X, y):
        print(f'\nTraining for Fold {fold}\n')

        model.fit(X[train_index], y[train_index], batch_size=BATCH_SIZE_LARGE_CNN_MODEL,
                  epochs=K_FOLD_EPOCHS, validation_split=K_FOLD_VALIDATION_SIZE, verbose=1, callbacks=[csv_logger])

        print("\nEvaluation\n")
        test_evaluation = model.evaluate(X[val_index], y[val_index], batch_size=EVALUATION_BATCH_SIZE, verbose=1)

        print("Test accuracy :{:5.2f}%".format(100 * test_evaluation[1]))
        fold_accuracy_values.append(test_evaluation[1] * 100)

        k.clear_session()
        fold += 1

    print("\nAccuracy values of folds\n", fold_accuracy_values)
    print("Mean: ", np.mean(fold_accuracy_values))
    print("Standard D.: ", np.std(fold_accuracy_values))

    return


def create_compile_tl_cnn_model(X_train, X_val, y_train, y_val):
    # include_top=False means fully connected layer's head will not be loaded for new input shape
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    for layer in vgg16.layers:
        layer.trainable = False

    head_model = vgg16.output

    head_model = Flatten(name="flatten")(head_model)

    head_model = Dense(256, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)

    head_model = Dense(len(CATEGORIES), activation="softmax")(head_model)

    model = Model(inputs=vgg16.input, outputs=head_model)

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    csv_logger = CSVLogger('training_log/log_tl_cnn_model.csv', append=False, separator=';')
    # early_stopping = EarlyStopping(monitor="val_loss", mode='auto', patience=5, verbose=1)
    # model_checkpoint = ModelCheckpoint(filepath="best_tl_cnn_model.h5", monitor="val_loss", verbose=1,
    # save_best_only=True)

    print("\nTraining:\n")
    model.fit(X_train, y_train, batch_size=BATCH_SIZE_TL_CNN_MODEL, epochs=EPOCHS_TL_CNN_MODEL, verbose=1,
              validation_data=(X_val, y_val), callbacks=[csv_logger])

    plot_evaluation(model)

    return model


def test_and_predict(model, X_test, y_test):
    print("\nTesting Model:\n")

    test_evaluation = model.evaluate(X_test, y_test, batch_size=EVALUATION_BATCH_SIZE, verbose=1)

    print("\nTest loss     :{:5.2f}%".format(100 * test_evaluation[0]))
    print("Test accuracy :{:5.2f}%".format(100 * test_evaluation[1]))

    predicted_classes = model.predict(X_test)

    y_pred = np.argmax(np.round(predicted_classes), axis=1)
    y_test = np.argmax(np.round(y_test), axis=1)

    main.correct_prediction_number(y_pred, y_test)

    print("\n", classification_report(y_test, y_pred, target_names=CATEGORIES))

    return


def plot_evaluation(model):
    acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    return


def extract_cnn_features(model, X_train, X_test):
    model_to_extract_features = Model(model.inputs, model.layers[-2].output)
    extracted_features_train = model_to_extract_features.predict(X_train)
    extracted_features_test = model_to_extract_features.predict(X_test)

    return extracted_features_train, extracted_features_test


def visualize_cnn_features(model):
    conv_layer_indexes = []

    for index in range(len(model.layers)):
        layer = model.layers[index]
        if 'conv' not in layer.name:
            continue
        conv_layer_indexes.append(index)

    output_of_layer = [model.layers[i].output for i in conv_layer_indexes]
    # output_of_layer = model.layers[conv_layer_indexes[-1]].output
    model_to_extract_features = Model(inputs=model.inputs, outputs=output_of_layer)

    # New Single Image for Feature Extraction
    # img = cv2.imread('dog.jpg', 0)
    # img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    # img = img.astype('float32')
    # img = img / 255
    # img = img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    # feature_outputs = model_to_extract_features.predict(img)

    # Visualization of Features
    # columns = 8
    # rows = 8
    # for feature in feature_outputs:
    #     fig = plt.figure(figsize=(12, 12))
    #     for i in range(1, columns * rows + 1):
    #         fig = plt.subplot(rows, columns, i)
    #         fig.set_xticks([])
    #         fig.set_yticks([])
    #         plt.imshow(feature[0, :, :, i - 1], cmap='gray')
    #     plt.show()

    train_data = []
    extracted_features = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_number = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                image = image.astype('float32')
                image = image / 255
                image = image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

                feature_outputs = model_to_extract_features.predict(image)
                extracted_features.append(feature_outputs)

                # plt.imshow(image, cmap="gray")
                # plt.show()

                # Visualization of Features
                # columns = 8
                # rows = 8
                # for feature in feature_outputs:
                #     fig = plt.figure(figsize=(12, 12))
                #     for i in range(1, columns * rows + 1):
                #         fig = plt.subplot(rows, columns, i)
                #         fig.set_xticks([])
                #         fig.set_yticks([])
                #         plt.imshow(feature[0, :, :, i - 1], cmap='gray')
                #     plt.show()

                train_data.append([image, class_number])
            except Exception as e:
                print(e)

    return extracted_features
