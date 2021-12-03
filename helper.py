#!/usr/bin/env python3

# helper needs to be in the same dir
# def s():
#     import importlib
#     importlib.reload(helper)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                                    ########################
                                    #  NAIVE BAYES         #
                                    #  RAW IMPLEMENTATION  #
                                    ########################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from collections import defaultdict

def label_indices(labels):
    """
    Group samples based on their labels and return indices
    @param labels: list of labels
    @return: dict, {class1: [indices], class2: [indices]}
    """
    dct = defaultdict(list)
    for index,label in enumerate(labels):
        dct[label].append(index)
    return dct

def prior(label_indices):
    """
    Compute prior based on training samples
    @param label_indices: grouped sample indices by class
    @return: dictionary, with class label as key, corresponding prior as value
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}

    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count

    return prior

def likelihood(features, label_indices, smoothing=0):
    """
    Compute likelihood based on training samples
    @param features: matrix of features
    @param label_indices: grouped sample indices by class
    @param smoothing: integer, additive smoothing parameter
    @return: dictionary, with class as key, corresponding conditional probability P(feature|class) vector as value
    """
    likelihood = {}

    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)

    return likelihood

def posterior(X, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and likelihood
    @param X: testing samples
    @param prior: dictionary, with class label as key, corresponding prior as value
    @param likelihood: dictionary, with class label as key, corresponding conditional probability vector as value
    @return: dictionary, with class label as key, corresponding posterior vector as value
    """

    posteriors = []
    for x in X:
        # posterior is proportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        # normalise so that all sums up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                                    ##################
                                    #  NAIVE BAYES   #
                                    #  SCIKIT-LEARN  #
                                    ##################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

from sklearn.naive_bayes import BernoulliNB

def nbayes(x_train, y_train, x_test):
    """
    Naive Bayes implementation using the scikit-learn library
    @param x_train: training samples
    @param y_train: training labels
    @param x_test: testing samples
    @return: nothing
    """

    # in sklearn alpha refers to the smoothing factor
    # fit_prior=True means that the prior will be learned form the training set
    clf = BernoulliNB(alpha=1.0, fit_prior=True)
    clf.fit(x_train, y_train)

    print(f'[sklearn] Predicted Probabilites: {clf.predict_proba(x_test)}')
    print(f'[sklearn] Predicted Class: {clf.predict(x_test)}')
