from sklearn import datasets
from sklearn.model_selection import train_test_split
from inference import BufferClassifier, BMM
import numpy as np
import pickle
from functions import relabel

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) // 9

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.85, shuffle=False
)
print("train size:", X_train.shape[0])
classifier = BufferClassifier(X_train.shape[0], 10, classify_dist="LL", merge_dist="Hellinger",
                              weight_scheme="linear")
labels = np.empty(X_test.shape[0], dtype=np.uint8)
for buffer in X_train:
    label = classifier.classify(buffer)
# save_model(classifier, "mnist_classifier")
classifier.save("mnist_classifier.pkl")
with open("mnist_classifier.pkl", "wb") as f:
    pickle.dump(X_train, f)
    pickle.dump(y_train, f)
for idx, buffer in enumerate(X_test):
    labels[idx] = classifier.classify(buffer)
y_predict = relabel(labels, 10, y_test)
# Even though the clustering problem is not our main focus, we still wish to evaluate the performance of our classifier.
# We compare the performance of our classifier with the performance of the BMM classifier.
# The BMM classifier is a simple classifier that uses the Bayesian Mixture Model to classify the buffers.
# The BMM classifier is implemented in the inference.py file.
# The BMM classifier is not used in the paper, but it is used here to compare the results with the clustering classifier.
