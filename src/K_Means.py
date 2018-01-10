"""
COMP 4107 Assignment #2

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466
"""

import numpy as np
import random
from math import inf

# create clusters and update centroids till optimal centroids found
def k_means_clustering(k, data, iterations):

    # initialize centroids and starting clusters
    centroids = init_centroids(k, data)
    clusters = create_clusters(data, centroids)

    # loop update->create->update till no more change can be made to the centroids
    old_difference = 0
    i = 0
    while i < iterations:
        old_centroids = centroids
        centroids = update_centroids(clusters)
        clusters = create_clusters(data, centroids)

        # find out how much our centroids moved
        num_centroids = len(centroids)
        differences = []
        for i in range(num_centroids):
            differences.append(np.linalg.norm(old_centroids[i] - centroids[i]))

        max_diff = max(differences)

        difference_change = abs((max_diff-old_difference)/np.mean([old_difference,max_diff])) * 100

        if np.isnan(difference_change):
            break
        i+=1

    return (centroids,clusters)


# pick k random centroids from the dataset
# use only the digit_vector, and not the label/target
def init_centroids(k, data):
    centroids = []

    # random.sample returns a list of k unique elements sampled from the dataset
    choices = random.sample(data, k)

    # select the digit_vector from the tuples in choices
    for i in range(k):
        centroids.append(choices[i][0])

    return centroids

# return the mean of the cluster
def cluster_mean(cluster):
    # sum up all digit_vectors in the cluster
    cluster_sum = cluster[0][0].copy()
    for x in cluster[1:]:
        cluster_sum += x[0]

    #  the mean (average) of the cluster
    cluster_mean = cluster_sum * (1.0 / len(cluster))

    return cluster_mean


# create clusters
# find the closest centroid to a datapoint and then assign the datapoint
# to the cluster around that centroid
def create_clusters(data, centroids):
    clusters = [[] for centroid in centroids]

    for x, y in data:
        smallest_distance = inf
        smallest_centroid_index = 0

        i = 0
        for centroid in centroids:
            distance = np.linalg.norm(x - centroid)
            if (distance < smallest_distance):
                smallest_centroid_index = i
                smallest_distance = distance
            i += 1

        clusters[smallest_centroid_index].append((x, y))

    return clusters

# update and move centroids to the mean of the cluster
def update_centroids(clusters):
    updated_centroids = []

    for cluster in clusters:
        updated_centroids.append(cluster_mean(cluster))

    return updated_centroids
