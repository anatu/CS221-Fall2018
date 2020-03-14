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
            # Initialize the weights for features if they aren't already in the weight vector 
            for feat in features.keys(): 
                if feat not in weights: weights[feat] = 0
            # Calculate the hinge loss
            score_product = (sum(weights[key]*features.get(key, 0) for key in weights))*y
            hingeLoss = max(0,1-score_product)
            print("Example: %s , Error: %i  " % (x, hingeLoss))
            # Compute gradient vector based on value of the hinge loss 
            if score_product < 1: # Equals phi*y if less than 1
                hingeGrad = features.update((a, b*y) for a, b in features.items())
            else: # Zero otherwise
                hingeGrad = 0
            
            # Update only if the gradient is nonzero, otherwise 
            # gradient descent cannot proceed
            if hingeGrad != 0:
                for feature in weights.keys():
                    weights[feature] = weights[feature] - eta*hingeGrad.get(feature, 0)

    # END_YOUR_CODE
    return weights

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
        raise Exception("Not implemented yet")
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
    for k in range(K):
    	centroids.append(random.choice(examples))

    # Initialize assignments list and loss storage
    assignments = [0 for elem in examples]
    assmts_lookup = {k: [] for k in range(centroids)}
    losses = dict()
    num_iters = 0
    while num_iters < maxIters: 
        num_iters += 1
        # Step 1 - Perform the initial assignment of examples to centroids
        for i in range(examples):
            distances = dict()
            for j in range(len(centroids)):
                distances[j] = sum((example.get(d,0) - centroids[j].get(d,0))**2 for d in set(X) | set(Y))
            assignments[i] = j
            # Add the example into the lookup table to speed up the update step
            assmts_lookup[j].append(example)
        # Step 2 - Update the centroids using the mean of assigned points
        for centroid in centroids:
            centroid = 


    	


    # END_YOUR_CODE

