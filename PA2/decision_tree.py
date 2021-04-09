# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
from collections import Counter
# Scikit learn is used only for drawing confusion matrix.
from sklearn.metrics import classification_report, confusion_matrix
# Matplotlib is used only for graph plotting
import matplotlib.pyplot as plt



def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')
    unique_values = np.unique(x)
    my_dict = {}
    for i in unique_values:
        my_dict[i] = np.where(x==i)
           
    return my_dict
    


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')

    # weighted_attributes = {0: 0, 1: 0}
    # for i in range(len(y)):
    #     if(y[i]==0):
    #         weighted_attributes[0] += weight[i]
    #     else:
    #         weighted_attributes[1] += weight[i]
    # total_sum=weighted_attributes[0]+weighted_attributes[1]
    # ent=[]
    # for i in range(len(weighted_attributes)):
    #     prob= weighted_attributes[i]/total_sum

    #     if prob != 0:
    #         logs=-np.log2(prob)
    #         ent.append(prob*logs)

    # return sum(ent)

    (_, counts) = np.unique(y, return_counts=True)
    total_vals = sum(counts)
    probs = counts/total_vals
    logs = -np.log2(probs)
    ents = probs*logs
    return sum(ents)



def mutual_information(x, y,weight):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')
    entY = entropy(y)

    # for binary split consider zeroes and ones
    
    hashmap = {}
    for i in np.unique(x):
        hashmap[i] = dict(Counter(np.array(y[np.where(x==i)]).flatten()))

    total_ent = 0   
    for j in hashmap:
        total_ent += entropy_calculation_from_dictionary(hashmap[j], len(y))

    return entY - total_ent


def entropy_calculation_from_dictionary(my_dict, total_labels):
    total_sum = sum(my_dict.values())
    res = 0
    for i in my_dict:
        frac = (my_dict[i]/total_sum)
        res -= frac * np.log2(frac)

    res = (total_sum/total_labels)*res
    return res



def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # raise Exception('Function not yet implemented!')
    unique_labels, count_unique_labels = np.unique(y, return_counts = True)
    
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(x.shape[1]):
            unique_attributes = partition(x[:, i])
            for each_attribute_from_set in unique_attributes.keys():
                attribute_value_pairs.append((i, each_attribute_from_set))
    attribute_value_pairs = np.array(attribute_value_pairs).astype(int)
    
    if len(unique_labels)==1:
        return unique_labels[0]
    
    if len(attribute_value_pairs)==0 or depth == max_depth:
        return unique_labels[np.argmax(count_unique_labels)]

    entropy_info = []

    for feature_column, value in attribute_value_pairs:
        indices = np.where(x[:, feature_column] == value)[0] 
        y_for_feature_single_attribute = y[indices] 
        entropy_info_for_feature_single_attribute = entropy(y_for_feature_single_attribute)
        entropy_info.append(entropy_info_for_feature_single_attribute)

    entropy_info_array = np.array(entropy_info)
    (max_attribute, max_value) = attribute_value_pairs[np.argmin(entropy_info_array)]
    max_attribute_partition = partition(np.array(x[:, max_attribute] == max_value).astype(int))
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(np.all(attribute_value_pairs == (max_attribute, max_value), axis=1)),0)

    decision_tree = {}

    for decision_value, indices in max_attribute_partition.items():
        x_new = x[indices]
        y_new = y[indices]
        attribute_decision = bool(decision_value)

        decision_tree[(max_attribute, max_value, attribute_decision)] = id3(x_new, y_new, attribute_value_pairs=attribute_value_pairs, max_depth=max_depth, depth=depth+1)

    return decision_tree
    


def bagging(x, y, max_depth, num_trees):
    """Input: x = Feature set, y = Labels, max depth=maximum depth of tree (more for bagging), num_trees = no. of bags
    Returns predictions set of different trees
    """

    predictions = {}
    alpha = 1

    length_of_data_set = len(x)
    # weights=np.ones(length_of_data_set)

    # loop through with number of bags and call id3 recursively to get the predicted tree
    for i in range(num_trees):
        # generate random indices of sample(bootstrap) with replacement
        bootstrap_indices = np.random.choice(length_of_data_set, size=length_of_data_set, replace=True)

        # apply id3 to get the prediction
        predicted_tree = id3(x[bootstrap_indices], y[bootstrap_indices], max_depth=max_depth)

        predictions[i] = (alpha, predicted_tree)

    return predictions


def boosting(x, y, max_depth, num_stumps):
    """Input: x = Feature set, y = Labels, max_depth = maximum depth of tree = 1, 2 (stumps/weak learner), num_trees = no. of bags
    Return predictions set of different trees
    """

    # length of training set
    length_of_data_set = len(x)

    # calculate weights of every example -> equal for first time
    weights = np.ones(length_of_data_set)/length_of_data_set
    predictions={}
    # initialize alpha values to 1
    alpha=1

    for i in range(num_stumps):
        # generate random sample after applying weights
        bootstrap_indices = np.random.choice(length_of_data_set, size=length_of_data_set, replace=True, p=weights)
        
        # get predicted tree
        predicted_tree=id3(x[bootstrap_indices],y[bootstrap_indices],max_depth=max_depth)
        
        # initialize the error and predicted labels array
        error = 0
        y_pred = []

        # calculate the error of every prediction and add only misclassifications
        for j in range(length_of_data_set):
            y_pred.append(predict_example(x[j, :], predicted_tree))
            if y_pred[j] != y[j]:
                error += weights[j]

        # calculate new alpha value
        alpha = 0.5 * (np.log(1-error) - np.log(error))

        # update the weights according to misclassifications -> exp(-alpha) for correct and vice versa
        for j in range(length_of_data_set):
            if y_pred[j] != y[j]:
                weights[j] = weights[j]*np.exp(alpha)
            else:
                weights[j] = weights[j]*np.exp(-alpha)


        # Normalize weights so that they add up to 1
        weights = weights/weights.sum()

        # add this prediction to final result
        predictions[i] = (alpha, predicted_tree)

    return predictions

def predict_boost_bag_example(x, h_ens):
    """Input: x = example for prediction, h_ens = dictionary of alpha and predicted trees
    """

    # initialize the avg of predictions to 0
    avg = 0

    for _, value in h_ens.items():
        # predict the example
        y_predicted=predict_example(x,value[1])

        # add product of alpha and predicted value
        avg += value[0]*y_predicted

    return 1 if avg > 0.5 else 0





def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # raise Exception('Function not yet implemented!')
    for attribute_keys, sub_tree in tree.items():
        attribute = attribute_keys[0]
        value = attribute_keys[1]
        decision = attribute_keys[2]

        if decision == (x[attribute] == value):
            if type(sub_tree) is dict:
                label = predict_example(x, sub_tree)
            else:
                label = sub_tree

            return label
    


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    # INSERT YOUR CODE HERE
    sum=0
    n=len(y_true)
    for x in range(n):
        if(y_true[x]!=y_pred[x]):
            sum= sum+1
    err=sum/n
    return err


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./data/mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]


    pred = boosting(Xtrn,ytrn,1,10)
    for i, value in pred.items():
        # visualize(value[1])
        y_pred = [predict_boost_bag_example(x, pred) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        print(tst_err*100)




    # # Learn a decision tree of depth 3
    # decision_tree = id3(Xtrn, ytrn, max_depth=1)
    # visualize(decision_tree)

    # # Compute the test error
    # y_pred = [predict_example(x, decision_tree) for x in Xtst]
    # tst_err = compute_error(ytst, y_pred)

    # print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    # train_error = compute_error(ytrn, y_pred)

    # print('Train Error = {0:4.2f}%.'.format(train_error * 100))

    # Draw Confusion Matrix with scikit
    # print()
    # print("Confusion Matrix")
    # print(confusion_matrix(ytst, y_pred))


    # Calculating average of training errors and testing errors - MONKS 1 Problem
    # for i in range(1, 4):
    #     filename_train = str('./monks-'+str(i)+'.train')
    #     M = np.genfromtxt(filename_train, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    #     ytrn = M[:, 0]
    #     Xtrn = M[:, 1:]


    #     filename_test = str('./monks-'+str(i)+'.test')
    #     M = np.genfromtxt(filename_test, missing_values=0, skip_header=0, delimiter=',', dtype=int)
    #     ytst = M[:, 0]
    #     Xtst = M[:, 1:]
    #     monks_test_error = []
    #     monks_train_error = []
    #     decision_tree_depths = []
    #     for j in range(1, 11):
    #         decision_tree = id3(Xtrn, ytrn, max_depth=j)
    #         y_pred_test = [predict_example(x, decision_tree) for x in Xtst]
    #         tst_err = compute_error(ytst, y_pred_test)

    #         y_pred_train = [predict_example(x, decision_tree) for x in Xtrn]
    #         train_error = compute_error(ytrn, y_pred_train)
    #         monks_test_error.append(tst_err)
    #         monks_train_error.append(train_error)
    #         decision_tree_depths.append(j)

        # fig,ax=plt.subplots()
        # plt.xlabel('Decision Tree Depth')
        # plt.ylabel('Errors')
        # plt.plot(decision_tree_depths,monks_train_error,'g',label='Training Error')
        # plt.plot(decision_tree_depths, monks_test_error,'p',label='Test Error', linestyle='dashed')
        # legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
        # plt.savefig('monks__'+ str(i)+ '.png')
        # plt.show()
        # print(train_error)
    