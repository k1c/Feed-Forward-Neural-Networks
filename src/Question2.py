"""
COMP 4107 Assignment #2

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

Question 2

Develop a feed forward RBF neural network in python that classifies the images found in the MNIST dataset.
You are to train your neural network using backpropagation.
You should use gaussian functions as your radial basis functions.
You must show that you have:

    Used K-means to design the hidden layer in your network.
    Performed K-fold cross correlation.
    Investigated the performance of your neural network for different sizes of hidden layer."""

import MNIST_Loader
import RBFNetwork
import K_Means

#loadMNISTData() returns a dataset which is a list of length 70,000 containing (image, label) tuples:
    #image is the input data in the form of a 784x1 vector where each element is a normalized greyscale value
    #label is the target data in the form of a 10x1 vector with a 1 in the index of the target value
training_dataset, testing_dataset = MNIST_Loader.load_data()
num_centroids = 16
num_iterations = 100
import random
random.shuffle(training_dataset)
print("K_Means Clustering in progress...")
final_centroids, final_clusters = K_Means.k_means_clustering(num_centroids, training_dataset[:1000].copy(),num_iterations)

print("LOOKK HERE")
print(len(final_centroids))
#hyper-parameters
learning_rate = 0.1
epochs = 10
num_folds = 5

scores = RBFNetwork.k_fold_cross_validation(training_dataset[:1000], num_folds, learning_rate, epochs, final_centroids, final_clusters)
print('\nMean Classification Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

print("\nTesting with MNIST testing dataset")
# initialize and train our network



layers = [len(final_centroids), 10]
network = RBFNetwork.Network(layers,final_centroids,final_clusters)
network.stochastic_gradient_descent(training_dataset[:1000].copy(), learning_rate, epochs)

print('\nNetwork Accuracy: %.3f%%' % ((network.network_accuracy(testing_dataset)/len(testing_dataset)) * 100))
