from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

# sigmoid activation function
def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

# compute the derivative of the sigmoid function
def sigmoid_deriv(x):
    return x * (1 - x)

def predict(X, W):
    # take the dot product between out features and weight matrix
    preds = sigmoid_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds

# construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1's as the last entry in the feature
# matrix -- this little trick allows us to treat the bias 
# as a trainable paramenter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO training...")
# initialize our weight matrix and list of losses
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    preds = sigmoid_activation(trainX.dot(W))
    error = preds - trainY
    # square function
    loss = np.sum(error ** 2)
    losses.append(loss)

    # the gradient descent update is the dot produce between
    # features and error of sigmoid derivative of predictions
    d = error * sigmoid_deriv(preds)
    gradient = trainX.T.dot(d)

    # nudge the weight matrix in the negative directiona of the gradient
    # hence the term "gradient descent" by taking a small step towards a 
    # set of "more optiomal" parameters
    W = W - args["alpha"] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch {}, loss {:.7f}".format(int(epoch + 1), loss))
    
# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classfication data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

