#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    features = dict()
    for word in set(x.split(" ")):
        features[word] = x.split(" ").count(word)
    return features
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for iteration in range(numIters):
        print("BEGINNING RUN NUMBER %i" % (iteration))
        for example in trainExamples:
            # Extract the features from the input
            x = example[0]
            y = example[1]
            features = featureExtractor(x)
            # Calculate the hinge loss
            score_product = (dotProduct(weights, features))*y
            hingeLoss = max(0,1-score_product)
            # Compute gradient vector based on value of the hinge loss 
            if score_product < 1: # Equals -phi*y if less than 1
                hingeGrad = features
                hingeGrad.update((a, b*-1*y) for a, b in features.items())
            else: # Zero otherwise
                hingeGrad = 0
            # Update weights for every training example (SGD)
            if hingeGrad != 0:
                for feature in hingeGrad.keys():
                    weights[feature] = weights.get(feature, 0) - eta*hingeGrad.get(feature)

                # Evaluate predictor performance
                # trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
                # print(trainError)

    return weights
    # END_YOUR_CODE

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {random.choice(weights.keys()): random.random()}
        y = 1 if dotProduct(phi, weights)>0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        # Preprocessing - ignore whitespace 
        x = x.replace(" ","")
        # Pull all ngrams in a zipped form and turn into a list of string grams
        ngram_tuple = zip(*[x[i:] for i in range(n)])
        ngram_list = ["".join(elem) for elem in ngram_tuple]
        features = dict()
        # Develop feature vector from grams
        for gram in ngram_list:
	        features[gram] = x.count(gram)
        return features
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    # Initialize the centroids, each one being a random selection
    # from the input list examples
    random.seed(42)
    centroids = []
    for k in range(K): centroids.append(random.choice(examples))
    
    loss_example_norm = 0
    assignments = [0 for elem in examples]
    num_iters = 0
    
    # LOSS - Example Norm
#     for example in examples:
#         loss_example_norm += sum((example.get(d))**2 for d in example)

    while num_iters < maxIters:
        num_iters += 1
        loss_cross_term9 = 0
        loss_cluster_norm = 0

        assmts_lookup = {k: [] for k in range(len(centroids))}
        # Declare array to store updated assignments, so that we can check for convergence 
        new_assignments = [0 for elem in examples]

        for i in range(len(examples)):
            distances = []
            for j in range(len(centroids)):
                # LOSS - Cluster Norm
#                 loss_cluster_norm += sum((centroids[j].get(d))**2 for d in centroids[j])
                # Calculate distance from each cluster 
                distances.append(sum((examples[i].get(key, 0) - centroids[j].get(key, 0))**2 for key in set(examples[i]) | set(centroids[j])))
            
            # Assign points to cluster
            z = distances.index(min(distances))
            new_assignments[i] = z
            assmts_lookup[z].append(examples[i])

        # Convergence Check
        if assignments == new_assignments:
            print "Converged after %i iterations" % (num_iters)
            reconstruction_loss = 0
            for k in range(len(centroids)):
                assignees = assmts_lookup[k]
                for assignee in assignees:
                    recr_loss = [(assignee.get(key, 0) - centroids[k].get(key, 0))**2 for key in set(assignee) | set(centroids[k])]
                    reconstruction_loss += sum(recr_loss)                

            return centroids, assignments, reconstruction_loss
        else: assignments = new_assignments
        
        # Calculate centroids and new reconstruction loss
        reconstruction_loss = 0
        new_centroids = [dict() for elem in centroids]
        for k in range(len(centroids)):
            assignees = assmts_lookup[k]
            for assignee in assignees:
                recr_loss = [(assignee.get(key, 0) - centroids[k].get(key, 0))**2 for key in set(assignee) | set(centroids[k])]
                reconstruction_loss += sum(recr_loss)
                # LOSS - Cross Term
#                 loss_cross_term += -2*dotProduct(centroids[k], assignee)
                # Assign new centroids
                for key in assignee.keys():
                    new_centroids[k][key] = new_centroids[k].get(key,0) + assignee.get(key)    
            new_centroids[k].update((a, b/float(len(assignees))) for a, b in new_centroids[k].items())
        centroids = new_centroids
        
    print("Maximum Iterations Reached")
    return centroids, assignments, reconstruction_loss
    # END_YOUR_CODE
    # END_YOUR_CODE