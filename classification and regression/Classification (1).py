import queue
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics


def gen_random_numbers_in_range(low, high, n):
    return random.sample(range(low, high), n)


# putting the side the extra reading for g class (balancing the dataset)
def balance_dataset(source, destination):
    list = gen_random_numbers_in_range(0, 12331, 5644)  # generate random numbers
    with open(source, 'r') as fp:
        lines = fp.readlines()  # read lines from the file
    with open(destination, 'w') as fp:
        # write the lines again in the destination file but without some random lines from list
        for number, line in enumerate(lines):
            if number not in list:
                fp.write(line)


# reading dataset from the data file
def reading_dataset(filename, X, y):
    with open(filename, 'r') as fp:
        line = fp.readline()
        i = 0
        while line:
            X.append(line.split(","))  # storing the features from the file in a list of lists (matrix)
            y.append(X[i][10])  # storing the label column in an array
            X[i].pop()
            X[i] = [float(x) for x in X[i]]  # convert string data to floating point
            i += 1
            line = fp.readline()


# storing split training, validation, test sets
def store_split_data(X_train, y_train, X_test, y_test, X_valid, y_valid):
    # storing test set data
    with open("test_set.data", 'w') as fp:
        for i in range(len(X_test)):
            for j in range(10):
                fp.write(str(X_test[i][j]))
                fp.write(",")
            fp.write(str(y_test[i]))

    # storing training set data
    with open("training_set.data", 'w') as fp:
        for i in range(len(X_train)):
            for j in range(10):
                fp.write(str(X_train[i][j]))
                fp.write(",")
            fp.write(str(y_train[i]))

    # storing validation set data
    with open("validation_set.data", 'w') as fp:
        for i in range(len(X_valid)):
            for j in range(10):
                fp.write(str(X_valid[i][j]))
                fp.write(",")
            fp.write(str(y_valid[i]))


# plotting the accuracy, precision, recall and f_score with the k values
def plotting(accuracy, precision, recall, f_score, k):
    plt.subplot(221)
    plt.plot(k, accuracy, marker='o')  # plotting accuracy with k values
    plt.xlabel("K Values")
    plt.ylabel("Accuracy")

    plt.subplot(222)
    plt.plot(k, precision, marker='o')  # plotting precision with k values
    plt.xlabel("K Values")
    plt.ylabel("Precision")

    plt.subplot(223)
    plt.plot(k, recall, marker='o')  # plotting recall with k values
    plt.xlabel("K Values")
    plt.ylabel("Recall")

    plt.subplot(224)
    plt.plot(k, f_score, marker='o')  # plotting f score with k values
    plt.xlabel("K Values")
    plt.ylabel("F Score")
    plt.show()


# apply knn classifier model with the best k on the test set
def model(best_k, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)  # training the model on the training set
    y_pred = knn.predict(X_test)  # predict values of y corresponding to values of X of the test set
    # scores of the best model
    accuracy = accuracy_score(y_test, y_pred)  # accuracy
    cm = confusion_matrix(y_test, y_pred)  # confusion matrix = [[TN FP] [FN TP]]
    precision = cm[1][1] / (cm[1][1] + cm[0][1])  # precision = TP / TP + FP
    recall = cm[1][1] / (cm[1][1] + cm[1][0])  # recall = TP / TP + FN
    f_score = 2 * (precision * recall) / (precision + recall)  # f score
    print('Accuracy:', accuracy, '\nPrecision:', precision, '\nRecall', recall, '\nF Score', f_score)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.show()


# applying knn classifier
def knn_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test):
    k = [i for i in range(1, 31)]
    precision = np.empty(len(k))
    accuracy = np.empty(len(k))
    recall = np.empty(len(k))
    f_score = np.empty(len(k))
    conf_mat = queue.Queue()
    # Hyperparameter (k) Tuning
    # by applying knn classifier on different k values on the validation set to detect the best model
    for i in range(len(k)):
        knn = KNeighborsClassifier(n_neighbors=k[i])
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_valid)  # predict the values of y corresponding to X values of the validation set
        # calculating different types of scores to detect the best k according to each
        accuracy[i] = accuracy_score(y_valid, y_pred)  # 1. accuracy
        cm = confusion_matrix(y_valid, y_pred)  # confusion matrix = [[TN FP] [FN TP]]
        conf_mat.put(cm)
        precision[i] = cm[1][1] / (cm[1][1] + cm[0][1])  # 2. precision = TP / TP + FP
        recall[i] = cm[1][1] / (cm[1][1] + cm[1][0])  # 3. recall = TP / TP + FN
        f_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])  # 4. f score

    # plotting the accuracy, precision, recall and f_score with the k values
    plotting(accuracy, precision, recall, f_score, k)

    # Data of the best 4 models
    best_k_a = accuracy.argmax() + 1  # best k according to accuracy
    print('\nData of Model_1 (best k according to the accuracy): k =', best_k_a)
    model(best_k_a, X_train, y_train, X_test, y_test)  # model of the best k according to accuracy

    best_k_p = precision.argmax() + 1  # best k according to precision
    print('\nData of Model_2 (best k according to the precision): k =', best_k_p)
    model(best_k_p, X_train, y_train, X_test, y_test)  # model of the best k according to precision

    best_k_r = recall.argmax() + 1  # best k according to recall
    print('\nData of Model_3 (best k according to the recall): k =', best_k_r)
    model(best_k_r, X_train, y_train, X_test, y_test)  # model of the best k according to recall

    best_k_f = f_score.argmax() + 1  # best k according to f_score
    print('\nData of Model_4 (best k according to the f score): k =', best_k_f)
    model(best_k_f, X_train, y_train, X_test, y_test)  # model of the best k according to f score


# splitting the dataset into training, validation and test sets
def split_dataset(X, y):
    train_ratio = 0.70
    valid_ratio = 0.15
    test_ratio = 0.15
    # first splitting X and y into training set 70% and the remaining 30%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - train_ratio, random_state=42)
    # then splitting the remaining X and Y into validation and dataset each 15%
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp,
                                                        test_size=test_ratio / (test_ratio + valid_ratio),
                                                        random_state=42)
    # asking the user if he wants to store the split datasets
    save = input('If you want to store the split training, validation and test datasets in separate files enter 1 '
                 'or enter 0 if not: ')
    if save == '1':
        store_split_data(X_train, y_train, X_test, y_test, X_valid, y_valid)
    # calling the knn classifier to start the classification process
    knn_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test)


# asking the user if he wants to balance the dataset
balance = input("If it is the first time to run the program enter 1 to balance dataset or enter 0 if not: ")
if balance == '1':
    balance_dataset("magic04.data", "magic04(2).data")  # balancing the dataset to have equal samples from both clases

X = []
y = []
reading_dataset("magic04(2).data", X, y)  # reading the dataset from the data file
split_dataset(X, y)  # splitting the dataset int training, validation and test sets
