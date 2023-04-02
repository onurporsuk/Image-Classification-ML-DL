from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from datetime import timedelta
from keras import backend as k
import ml_classification
import tensorflow as tf
import numpy as np
import cnn_models
import winsound
import random
import time
import cv2
import os


DATADIR = ""
CATEGORIES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
IMAGE_SIZE = 100

TEST_SIZE = 0.1  # Splitting dataset into train and test with specified portion
VALIDATION_SIZE = 0.5  # Splitting test data to test and validation with specified portion


def read_data():
    data = []
    counter = {'COVID': 0, 'Lung_Opacity': 0, 'Normal': 0, 'Viral Pneumonia': 0}

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_number = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

                # plt.imshow(new_image, cmap="gray")
                # plt.show()

                data.append([new_image, class_number])
                counter[category] += 1

            except Exception as e:
                print(e)

    plt.figure(figsize=(5, 5))
    plt.bar(x=counter.keys(), height=counter.values())
    plt.show()

    random.shuffle(data)

    # Separating features and label from data
    X_features = []
    y_labels = []
    for features, labels in data:
        X_features.append(features)
        y_labels.append(labels)

    # Saving train data not to read image files over and over again
    np.save('numpy_files/images.npy', X_features)
    np.save('numpy_files/labels.npy', y_labels)

    return


def prepare_data():
    # Loading train data and splitting into train, test and validation
    features = np.load('numpy_files/images.npy')
    labels = np.load('numpy_files/labels.npy')

    # Printing some information of data
    print("\n\nDataset\n")
    print("\nOriginal data shape     :", features.shape, labels.shape)
    print('Output classes          :', np.unique(labels))

    # Data preprocessing
    features = features.astype('float32')
    features = features / 255
    features = features.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    return features, labels


def run_small_cnn_model():
    print("\n\n", "\t" * 6, "Small CNN Model\n\n")
    start_time_small_cnn_model = time.monotonic()

    small_cnn_model = cnn_models.create_compile_small_cnn_model(train_X, val_X, train_y_one_hot, val_y_one_hot)
    # small_cnn_model.save('saved_models/small_cnn_model')

    # small_cnn_model = keras.models.load_model('saved_models/small_cnn_model')
    # small_cnn_model.summary()

    cnn_models.test_and_predict(small_cnn_model, test_X, test_y_one_hot)

    end_time_small_cnn_model = time.monotonic()
    time_measurements[1] = timedelta(seconds=end_time_small_cnn_model - start_time_small_cnn_model)
    print("Small CNN Model execution time  :", time_measurements[1], "seconds")

    print("\n\nFeature Extraction\n")
    cnn_features_train, cnn_features_test = cnn_models.extract_cnn_features(small_cnn_model, train_X, test_X)
    print("CNN Small model features (train and test) shape:", cnn_features_train.shape, cnn_features_test.shape)

    k.clear_session()

    return cnn_features_train, cnn_features_test


def run_large_cnn_model():
    print("\n\n", "\t" * 6, "Large CNN Model\n\n")
    start_time_large_cnn_model = time.monotonic()

    # Applying K-Fold Cross Validation
    # cnn_models.k_fold_validate(X, y)

    large_cnn_model = cnn_models.compile_large_cnn_model(train_X, val_X, train_y_one_hot, val_y_one_hot)
    # large_cnn_model.save('saved_models/large_cnn_model')
    #
    # large_cnn_model = keras.models.load_model('saved_models/large_cnn_model')
    # large_cnn_model.summary()

    cnn_models.test_and_predict(large_cnn_model, test_X, test_y_one_hot)

    end_time_large_cnn_model = time.monotonic()
    time_measurements[2] = timedelta(seconds=end_time_large_cnn_model - start_time_large_cnn_model)
    print("Large CNN Model execution time  :", time_measurements[2], "seconds")

    print("\n\nCNN Model Feature Extraction\n")
    cnn_features_train, cnn_features_test = cnn_models.extract_cnn_features(large_cnn_model, train_X, test_X)
    print("CNN Large model features (train and test) shape:", cnn_features_train.shape, cnn_features_test.shape)

    # Saving extracted CNN features with corresponding labels
    # np.save('numpy_files/train_cnn_features.npy', cnn_features_X_train)
    # np.save('numpy_files/test_cnn_features.npy', cnn_features_X_test)

    k.clear_session()

    return cnn_features_train, cnn_features_test


def run_tl_cnn_model():
    print("\n\n", "\t" * 6, "Transfer Learning CNN Model - VGG16\n\n")
    start_time_tl_cnn_model = time.monotonic()

    # Converting grayscale images to RGB for pre-trained model VGG16
    train_X_rgb = np.repeat(train_X, 3, -1)
    val_x_rgb = np.repeat(val_X, 3, -1)
    test_X_rgb = np.repeat(test_X, 3, -1)

    print("New shape of training, validation and test data after RGB conversion:")
    print(train_X_rgb.shape)
    print(val_x_rgb.shape)
    print(test_X_rgb.shape)

    tl_cnn_model = cnn_models.create_compile_tl_cnn_model(train_X_rgb, val_x_rgb, train_y_one_hot, val_y_one_hot)
    # tl_cnn_model.save('saved_models/tl_cnn_model')

    # tl_cnn_model = keras.models.load_model('saved_models/tl_cnn_model')
    # tl_cnn_model.summary()

    cnn_models.test_and_predict(tl_cnn_model, test_X_rgb, test_y_one_hot)

    end_time_tl_cnn_model = time.monotonic()
    time_measurements[3] = timedelta(seconds=end_time_tl_cnn_model - start_time_tl_cnn_model)
    print("T.L. CNN Model execution time   :", time_measurements[3], "seconds")

    print("\n\nCNN Model Feature Extraction\n")
    cnn_features_train, cnn_features_test = cnn_models.extract_cnn_features(tl_cnn_model, train_X, test_X)
    print("CNN TL model features (train and test) shape:", cnn_features_train.shape, cnn_features_test.shape)

    k.clear_session()

    return cnn_features_train, cnn_features_test


def classify_cnn_features(cnn_features_train, cnn_features_test):
    print("\n\nCNN-SVM Hybrid Classification with CNN Features\n")
    ml_classification.svm_classifier(cnn_features_train, cnn_features_test, train_y_int, test_y_int)

    print("\nCNN-XGBoost Hybrid Classification with CNN Features\n")
    ml_classification.xgboost_classifier(cnn_features_train, cnn_features_test, train_y_int, test_y_int)

    return


# def classify_combined_features():
#     trad_features_X_train = np.load('numpy_files/train_traditional_features.npy')
#     trad_features_X_test = np.load('numpy_files/test_traditional_features.npy')
#
#     combined_features_X_train = np.concatenate((cnn_features_X_train, trad_features_X_train), axis=1)
#     combined_features_X_test = np.concatenate((cnn_features_X_test, trad_features_X_test), axis=1)
#
#     print("Shape of CNN features                  :", cnn_features_X_train.shape, cnn_features_X_test.shape)
#     print("Shape of traditional methods' features :", trad_features_X_train.shape, trad_features_X_test.shape)
#     print("Shape of combined features             :", combined_features_X_train.shape, combined_features_X_test.shape)
#
#     print("\n\nCNN-SVM Hybrid Classification with Traditional Methods' Features\n")
#     ml_classification.svm_classifier(trad_features_X_train, trad_features_X_test, train_y_int, test_y_int)
#
#     print("\nCNN-XGBoost Hybrid Classification with Traditional Methods' Features\n")
#     ml_classification.xgboost_classifier(trad_features_X_train, trad_features_X_test, train_y_int, test_y_int)
#
#     print("\n\nCNN-SVM Hybrid Classification with Combined Features\n")
#     ml_classification.svm_classifier(combined_features_X_train, combined_features_X_test, train_y_int, test_y_int)
#
#     print("\nCNN-XGBoost Hybrid Classification with Combined Features\n")
#     ml_classification.xgboost_classifier(combined_features_X_train, combined_features_X_test, train_y_int, test_y_int)
#
#     return


def correct_prediction_number(y_pred, y_test):
    correct = []
    incorrect = []
    for index in range(len(y_pred)):
        if y_pred.item(index) == y_test.item(index):
            correct.append(y_pred.item(index))
        else:
            incorrect.append(y_pred.item(index))

    print("\nCorrect predictions number   :", len(correct))
    print("Incorrect predictions number :", len(incorrect))

    return


def check_gpu():
    print("\nHardware Information for Computation\n")
    gpu_available = tf.config.list_physical_devices('GPU')
    if gpu_available is not None:
        print("Available hardware to compute is:", gpu_available)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(gpus[0],
                                                           [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

            except RuntimeError as e:
                print(e)
    else:
        print("\nAvailable hardware to compute is CPU")

    return


if __name__ == '__main__':
    # List to store all operations execution time
    time_measurements = [0, 0, 0, 0, 0]

    # Measurement of program execution time
    start_time_total = time.monotonic()

    # Measurement of reading and preparation data time
    start_time_read = time.monotonic()

    # Checking GPU availability
    check_gpu()

    # Data is already read and saved, this line below is commented
    # read_data()

    # Preparing dataset
    X, y = prepare_data()

    # Data splitting for CNN Models
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=TEST_SIZE)
    test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=VALIDATION_SIZE)

    # Saving splitted data
    # np.save('numpy_files/split/train_X.npy', train_X)
    # np.save('numpy_files/split/test_X.npy', test_X)
    # np.save('numpy_files/split/train_y.npy', train_y)
    # np.save('numpy_files/split/test_y.npy', test_y)
    # np.save('numpy_files/split/val_X.npy', val_X)
    # np.save('numpy_files/split/val_y.npy', val_y)

    # Loading splitted data
    # train_X = np.load('numpy_files/split/train_X.npy')
    # test_X = np.load('numpy_files/split/test_X.npy')
    # train_y = np.load('numpy_files/split/train_y.npy')
    # test_y = np.load('numpy_files/split/test_y.npy')
    # val_X = np.load('numpy_files/split/val_X.npy')
    # val_y = np.load('numpy_files/split/val_y.npy')

    # Label encoding instead of one-hot for Machine Learning methods
    train_y_int = train_y
    test_y_int = test_y

    # Printing some information of dataset
    print("\nDataset after preprocessing:")
    print("\nOriginal label              :", train_y[0])

    # One-hot encoding conversion
    train_y_one_hot = to_categorical(train_y)
    val_y_one_hot = to_categorical(val_y)
    test_y_one_hot = to_categorical(test_y)

    print("After conversion to one-hot :", train_y_one_hot[0])
    print("\nTraining data shape     :", train_X.shape, train_y_one_hot.shape)
    print("Validation data shape   :", val_X.shape, val_y_one_hot.shape)
    print("Test data shape         :", test_X.shape, test_y_one_hot.shape)

    end_time_read = time.monotonic()
    time_measurements[0] = timedelta(seconds=end_time_read - start_time_read)
    print("\nReading and preparing data time :", time_measurements[0], "seconds")

    # Running Small CNN Model
    # cnn_features_X_train, cnn_features_X_test = run_small_cnn_model()
    # classify_cnn_features(cnn_features_X_train, cnn_features_X_test)

    # Running Large CNN Model
    cnn_features_X_train, cnn_features_X_test = run_large_cnn_model()
    # cnn_features_X_train = np.load('numpy_files/train_cnn_features.npy')
    # cnn_features_X_test = np.load('numpy_files/test_cnn_features.npy')
    classify_cnn_features(cnn_features_X_train, cnn_features_X_test)
    # classify_combined_features()

    # Running TL CNN Model
    # cnn_features_X_train, cnn_features_X_test = run_tl_cnn_model()
    # classify_cnn_features(cnn_features_X_train, cnn_features_X_test)

    end_time_total = time.monotonic()
    time_measurements[4] = timedelta(seconds=end_time_total - start_time_total)
    print("Total execution time            :", time_measurements[4], "seconds")

    winsound.Beep(440, 2000)
