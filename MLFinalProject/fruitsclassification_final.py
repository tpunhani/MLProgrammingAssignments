
#Import statatements

import os #for accessing the file system
import numpy as np
import matplotlib.pyplot as plt

#skimage will be used for the conversion of image data
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle



# define a method to get all the fruit images which we want to extract
def getFruits(imageSize, data_path, fruitsCategory=[]):
    '''
    To convert passed fruit images to 100x100 size and return training and testing data
    '''

    if(len(fruitsCategory) == 0):
        for f_path in os.listdir(data_path):
            fruitsCategory.append(f_path)

    # Now we have all the folder names of our data inside fruitsCategory variable
    # We can go through every image and extract them according to count


    y_train = []
    images = []
    X_train = []



    for category in fruitsCategory:
        class_num = fruits.index(category)  # Converting labels into indices
        path = os.path.join(data_path, category)
        for img in os.listdir(path):
            if img.endswith('.jpg'):
                try:
                    # print the image path to track progress of training
                    print(os.path.join(path, img))

                    img_array = imread(os.path.join(path, img))

                    # resizing images to half of it's actual size for better performance of the model
                    img_resize = resize(img_array, (imageSize, imageSize, 3))  # Normalizes values between 0 and 1
                    X_train.append(img_resize.flatten())   # Add image data after flattening (1 D)
                    images.append(img_resize)
                    y_train.append(class_num)

                except (SystemError, IOError) as e:
                    print('BAD FILE - only jpg file types are accepted', img)

            

    
    # Convert the data into np array for further processing
    y_train = np.array(y_train)
    images = np.array(images)
    X_train = np.array(X_train)
    
    result = (X_train, y_train, images)

    return result



# Method to plot the images after resize
def plot_image_grid(images, nb_rows, nb_cols, figsize=(5, 5)):
    assert len(images) == nb_rows*nb_cols, "Number of images should be the same as (nb_rows*nb_cols)"
    fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_cols, figsize=figsize)
    for idx, image in enumerate(images):
        row = idx // nb_cols
        col = idx % nb_cols
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()



# Apply Principle Component Analysis to reduce the number of features (PCA)

# Normalize the data first so that it can be applied to PCA
def applyPCA(X_train, n_components):
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(images_scaled)

    return pca_result



def applyTSNE(pca_result, n_components):
    # apply TSNE
    tsne = TSNE(n_components=n_components, perplexity=40.0)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

    return tsne_result_scaled



def plotPCAResult(y_train, pca_result):
    plt.figure(figsize=(8,6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_train, cmap='plasma')
    plt.xlabel('First Principal component')
    plt.ylabel('Second Principal component')




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=fruits, yticklabels=fruits,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm,ax



def draw_confusion_matrix(clf, X_train, X_test, y_train, y_test, label='Confusion Matrix'):
    # y_score_tsne = clf.fit(X_train_tsne, y_train_tsne).predict(X_test_tsne)
    clf_1 = clf.fit(X_train, y_train)
    y_pred = clf_1.predict(X_test)
    accuracy_score(y_pred, y_test)


    #Evaluation
    precision = metrics.accuracy_score(y_pred, y_test) * 100
    print("Accuracy : {0:.2f}%".format(precision))
    cm , _ = plot_confusion_matrix(y_test, y_pred,classes=y_train, normalize=True, title=label)
    plt.show()



def draw_ROC_curve(clf, X_train, X_test, y_train, y_test, data_with_labels, label='Receiver operating characteristic'):
    y_train_binary = label_binarize(y_train, classes=np.array(list(data_with_labels.keys())))
    y_test_binary = label_binarize(y_test, classes=np.array(list(data_with_labels.keys())))

    y_score_binary = clf.fit(X_train, y_train_binary).predict(X_test)

    n_classes = y_train_binary.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_binary[:, i], y_score_binary[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_binary.ravel(), y_score_binary.ravel())
    # roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # #ROC curve for a specific class here for the class 2
    # auc = roc_auc[3]

    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label)
    plt.legend(loc="lower right")
    plt.show()

    result = (fpr[3], tpr[3], roc_auc[3])
    return result


def get_ROC_data_SVM(clf, X_train, X_test, y_train, y_test, y_test_binary, label='Receiver operating characteristic'):
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    
    n_classes = y_test_binary.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label)
    plt.legend(loc="lower right")
    plt.show()

    result = (fpr[3], tpr[3], roc_auc[3])
    return result



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
    


def entropy(y, weight):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')


    # for binary splits consider only 0 and 1 case
    weighted_attributes = {0: 0, 1: 0}
    for i in range(len(y)):
        if(y[i]==0):
            weighted_attributes[0] += weight[i]
        else:
            weighted_attributes[1] += weight[i]
    total_sum=weighted_attributes[0]+weighted_attributes[1]
    ent=[]
    for i in range(len(weighted_attributes)):
        prob = 0
        if total_sum != 0:
            prob= weighted_attributes[i]/total_sum

        if prob != 0:
            logs=-np.log2(prob)
            ent.append(prob*logs)

    return sum(ent)



def mutual_information(x, y, weight):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    # INSERT YOUR CODE HERE
    # raise Exception('Function not yet implemented!')
    entY = entropy(y, weight)


    unique_values_x, counts = np.unique(x, return_counts=True)

    probabilities_x = counts/len(x)

    mapping_of_probabilities = zip(probabilities_x,unique_values_x)
    
    #weighted-average entropy of each possible split
    for prob, unique_value in mapping_of_probabilities:
        entY-=prob*entropy(y[x==unique_value], weight) 
        
    return entY



def id3(x, y, weight, attribute_value_pairs=None, depth=0, max_depth=5):
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
    mutual_information_list = []

    for feature_column, value in attribute_value_pairs:
        indices = np.where(x[:, feature_column] == value)[0] 
        y_for_feature_single_attribute = y[indices] 
        entropy_info_for_feature_single_attribute = entropy(y_for_feature_single_attribute, weight)
        entropy_info.append(entropy_info_for_feature_single_attribute)
        mutual_information_list.append(mutual_information(x[:, feature_column], y, weight))

    # convert it into np array to find the argmax
    mutual_info_array = np.array(mutual_information_list, dtype=float)
    
    (max_attribute, max_value) = attribute_value_pairs[np.argmax(mutual_info_array)]
    max_attribute_partition = partition(np.array(x[:, max_attribute] == max_value).astype(int))
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(np.all(attribute_value_pairs == (max_attribute, max_value), axis=1)),0)

    decision_tree = {}

    for decision_value, indices in max_attribute_partition.items():
        x_new = x[indices]
        y_new = y[indices]
        attribute_decision = bool(decision_value)

        decision_tree[(max_attribute, max_value, attribute_decision)] = id3(x_new, y_new, weight, attribute_value_pairs=attribute_value_pairs, max_depth=max_depth, depth=depth+1)

    return decision_tree
    


def bagging(x, y, max_depth, num_trees):
    """Input: x = Feature set, y = Labels, max depth=maximum depth of tree (more for bagging), num_trees = no. of models, bag_size = no. of bags
    Returns predictions set of different trees
    """

    predictions = {}
    alpha = 1

    length_of_data_set = len(x)

    weight = np.ones(length_of_data_set)

    # loop through with number of bags and call id3 recursively to get the predicted tree
    for i in range(num_trees):
        # generate random indices of sample(bootstrap) with replacement
        bootstrap_indices = np.random.choice(length_of_data_set, size=length_of_data_set, replace=True)

        # apply id3 to get the prediction
        predicted_tree = id3(x[bootstrap_indices], y[bootstrap_indices], weight, max_depth=max_depth)

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
    # print(weights)
    predictions={}
    # initialize alpha values to 1
    alpha=1

    for i in range(num_stumps):
        # generate random sample after applying weights
        bootstrap_indices = np.random.choice(length_of_data_set, size=length_of_data_set, replace=True)
        
        # get predicted tree with new weights
        predicted_tree=id3(x[bootstrap_indices],y[bootstrap_indices], weights, max_depth=max_depth)
        
        
        # define the predicted values array
        y_pred = []
        
        # initialize no. of misclaffications
        mis_count = 0

        # calculate the error of every prediction and add only misclassifications
        for j in range(length_of_data_set):
            y_pred.append(predict_example(x[j, :], predicted_tree))
            if y_pred[j] != y[j]:
                mis_count += 1


        # calculate the avg error
        error = mis_count/length_of_data_set

        
        # calculate new alpha value
        alpha = 0.5 * (np.log((1-error)/error))


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

    # initialize the avg of predictions to 0, it will used for majoirity voting
    average_alpha = []

    # initialize the sum_alpha which will be used to calculate the avg of alphas
    sum_alpha = 0

    for _, model in h_ens.items():
        # model[0] = alpha, model[1] = hypothesis

        # predict the example
        y_predicted=predict_example(x,model[1])

        average_alpha.append(y_predicted*model[0])
        sum_alpha += model[0]

    predicted_example = np.sum(average_alpha)/sum_alpha


    return predicted_example >= 0.5



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

    err = sum/n

    return err



if __name__ == '__main__':

    # Get all the fruits and vegetables images from the dataset
    # Total images ~90k

    # DATA PREPROCESSING

    #Extract images on the basis of number of fruits and number of files of each category.
    # We will extract multiple images for multiclass classification


    path_of_data = 'fruits-360/Training/'
    path_of_data_test = 'fruits-360/Test/'


    # We can consider more classes but for now we will run our algorithms on 5 fruits
    fruits = ['Apple Red 1', 'Banana', 'Blueberry', 'Cocos', 'Corn']

    # define the image pixel size for which we want to run our algorithms
    imageSize = 50

    # Original training and test data
    X_train, y_train, images = getFruits(imageSize, path_of_data, fruitsCategory=fruits)
    X_test, y_test, _ = getFruits(imageSize, path_of_data_test, fruitsCategory=fruits)


    # Verify the shape of the data => number_of_features = imageSize*imageSize*3
    print("The shape of the original dataset" + str(X_train.shape))


    # print the labels which are passed for classification
    data_with_labels = {i: v for i, v in enumerate(fruits) }
    print("Images with labels" +str(data_with_labels))


    # Show the bar graph for all different labels
    unique, count = np.unique(y_train, return_counts=True)
    print("Drawing bar graph of the fruits")
    plt.bar(fruits, count)


    # Draw first 100 images of the data
    print("Drawing image grid for first 100 images")
    plot_image_grid(images[0:100], 10, 10)


    # Draw some other images of the data
    print("Drawing image grid graph for 1000-1400 images")
    plot_image_grid(images[1000:1400], 20, 20, figsize=(10,10))

    # apply PCA to convert the data into 2 dimensions, this will first normalize the data and than reduce the dimensionality
    print("Applying PCA and converting dataset into 2 features")
    pca = applyPCA(X_train=X_train, n_components=2)

    print("Drawing data scatter graph after applying pca")
    plotPCAResult(y_train, pca)

    # Check the dimention of new dataset
    print("The shape of the transformed dataset after pca"+str(pca.shape))


    # Apply PCA first and than use TSNE to get more accurate results
    print("Applying PCA and TSNE")
    pca_result = applyPCA(X_train, 50)
    tsne = applyTSNE(pca_result, 2)


    print("Drawing data scatter graph after applying pca+TSNE")
    plotPCAResult(y_train, tsne)


    # Split the data from test split function from scikit
    X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(tsne, y_train, test_size=0.25, random_state=42)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pca, y_train, test_size=0.25, random_state=42)
    X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


    # Run Decision Tree algorithm on depth = 3
    print("Classifying decision tree for depth = 3")
    decision_tree_clf = tree.DecisionTreeClassifier(random_state=42, max_depth=3)
    print("Drawing confusion matrix and ROC curves with original data")
    draw_confusion_matrix(decision_tree_clf, X_train, X_test, y_train, y_test, label='Confusion Matrix Decision Tree (Depth=3)')
    (fpr_test_dt, tpr_test_dt, auc_test_dt) = draw_ROC_curve(decision_tree_clf, X_train, X_test, y_train, y_test, data_with_labels, label='ROC Decision Tree (Depth=3)')


    print("Drawing confusion matrix and ROC curves with TSNE data")
    draw_confusion_matrix(decision_tree_clf, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, label='Confusion Matrix TSNE Decision Tree (Depth=3)')
    (fpr_tsne_dt, tpr_tsne_dt, auc_tsne_dt) = draw_ROC_curve(decision_tree_clf, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, data_with_labels, label='ROC Decision Tree TSNE (Depth=3)')
    
    print("Drawing confusion matrix and ROC curves with PCA data")
    draw_confusion_matrix(decision_tree_clf, X_train_pca, X_test_pca, y_train_pca, y_test_pca, label='Confusion Matrix PCA Decision Tree (Depth=3)')
    (fpr_pca_dt, tpr_pca_dt, auc_pca_dt) = draw_ROC_curve(decision_tree_clf, X_train_pca, X_test_pca, y_train_pca, y_test_pca, data_with_labels, label='ROC Decision Tree PCA (Depth=3)')


    print("Drawing confusion matrix and ROC curves with train test split data")
    draw_confusion_matrix(decision_tree_clf, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, label='Confusion Matrix WITHOUT PCA Decision Tree (Depth=3)')
    (fpr_without_pca_dt, tpr_without_pca_dt, auc_without_pca_dt) = draw_ROC_curve(decision_tree_clf, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, data_with_labels, label='ROC Decision Tree Train Test Split (Depth=3)')


    print("Finding the best k value for Nearest Neighbours using cross validation")
    # creating odd list of K for KNN
    neighbors = list(range(1, 50, 2))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    print("Performing 10 fold cross validation")
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())


    # changing to misclassification error
    mse = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[mse.index(min(mse))]
    print("The optimal number of neighbors is {}".format(optimal_k))

    # plot misclassification error vs k
    plt.plot(neighbors, mse)
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Misclassification Error")
    plt.show()


    # Run KNN algorithm on k = 1
    print("Running KNN algorithm for optimal k value")
    knn_clf = KNeighborsClassifier(n_neighbors=optimal_k)
    print("Drawing confusion matrix and ROC curves with original data")
    draw_confusion_matrix(knn_clf, X_train, X_test, y_train, y_test, label='Confusion Matrix KNN (Optimal K='+str(optimal_k)+')')
    (fpr_test_knn, tpr_test_knn, auc_test_knn) = draw_ROC_curve(knn_clf, X_train, X_test, y_train, y_test, data_with_labels, label='ROC KNN (Optimal K='+str(optimal_k)+')')


    print("Drawing confusion matrix and ROC curves with TSNE data")
    draw_confusion_matrix(knn_clf, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, label='Confusion Matrix TSNE KNN (Optimal K='+str(optimal_k)+')')
    (fpr_tsne_knn, tpr_tsne_knn, auc_tsne_knn) = draw_ROC_curve(knn_clf, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, data_with_labels, label='ROC KNN TSNE (Optimal K='+str(optimal_k)+')')
    
    print("Drawing confusion matrix and ROC curves with PCA data")
    draw_confusion_matrix(knn_clf, X_train_pca, X_test_pca, y_train_pca, y_test_pca, label='Confusion Matrix PCA KNN (Optimal K='+str(optimal_k)+')')
    (fpr_pca_knn, tpr_pca_knn, auc_pca_knn) = draw_ROC_curve(knn_clf, X_train_pca, X_test_pca, y_train_pca, y_test_pca, data_with_labels, label='ROC KNN PCA (Optimal K='+str(optimal_k)+')')


    print("Drawing confusion matrix and ROC curves with train test split data")
    draw_confusion_matrix(knn_clf, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, label='Confusion Matrix WITHOUT PCA KNN (Optimal K='+str(optimal_k)+')')
    (fpr_without_pca_knn, tpr_without_pca_knn, auc_without_pca_knn) = draw_ROC_curve(knn_clf, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, data_with_labels, label='ROC KNN Train Test Split (Optimal K='+str(optimal_k)+')')




    # Run ANN algorithm
    print("Running MLP - ANN algorithm on 2 hidden layers and 500 iterations")
    clf_mlp = MLPClassifier(hidden_layer_sizes=(200,200), activation='relu', solver='adam', max_iter=500,random_state=42)
    print("Drawing confusion matrix and ROC curves with original data")
    draw_confusion_matrix(knn_clf, X_train, X_test, y_train, y_test, label='Confusion Matrix MLP (Layers=2)')
    (fpr_test_knn, tpr_test, auc_test) = draw_ROC_curve(knn_clf, X_train, X_test, y_train, y_test, data_with_labels, label='ROC MLP (Layers=2)')


    print("Drawing confusion matrix and ROC curves with TSNE data")
    draw_confusion_matrix(knn_clf, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, label='Confusion Matrix TSNE MLP (Layers=2)')
    (fpr_tsne, tpr_tsne, auc_tsne) = draw_ROC_curve(knn_clf, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, data_with_labels, label='ROC MLP TSNE (Layers=2)')
    
    print("Drawing confusion matrix and ROC curves with PCA data")
    draw_confusion_matrix(knn_clf, X_train_pca, X_test_pca, y_train_pca, y_test_pca, label='Confusion Matrix PCA MLP (Layers=2)')
    (fpr_pca, tpr_pca, auc_pca) = draw_ROC_curve(knn_clf, X_train_pca, X_test_pca, y_train_pca, y_test_pca, data_with_labels, label='ROC MLP PCA (Layers=2)')


    print("Drawing confusion matrix and ROC curves with train test split data")
    draw_confusion_matrix(knn_clf, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, label='Confusion Matrix WITHOUT PCA MLP (Layers=2)')
    (fpr_without_pca, tpr_without_pca, auc_without_pca) = draw_ROC_curve(knn_clf, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, data_with_labels, label='ROC MLP Train Test Split (Layers=2)')



    print("Bagging classification on best depth = 4")
    dtc = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf_bagging=BaggingClassifier(base_estimator=dtc, n_estimators=15, bootstrap=True, verbose=True, max_samples=10, random_state=42)
    
    print("Drawing confusion matrix for original data")
    draw_confusion_matrix(clf_bagging, X_train, X_test, y_train, y_test, label='Confusion Matrix Bagging (DT depth=4)')

    print("Drawing confusion matrix for TSNE data")
    draw_confusion_matrix(clf_bagging, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, label='Confusion Matrix Bagging TSNE (DT depth=4)')

    print("Drawing confusion matrix for PCA data")
    draw_confusion_matrix(clf_bagging, X_train_pca, X_test_pca, y_train_pca, y_test_pca, label='Confusion Matrix Bagging PCA (DT depth=4)')

    print("Drawing confusion matrix for train test data")
    draw_confusion_matrix(clf_bagging, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, label='Confusion Matrix Bagging Train test split (DT depth=4)')


    print("Boosting classification on best depth = 2")
    dtc = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_boosting = AdaBoostClassifier(base_estimator=dtc, n_estimators=2, learning_rate=1, random_state=42)
    
    print("Drawing confusion matrix for original data")
    draw_confusion_matrix(clf_boosting, X_train, X_test, y_train, y_test, label='Confusion Matrix Boosting (DT depth=2)')

    print("Drawing confusion matrix for TSNE data")
    draw_confusion_matrix(clf_boosting, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, label='Confusion Matrix Boosting TSNE (DT depth=2)')

    print("Drawing confusion matrix for PCA data")
    draw_confusion_matrix(clf_boosting, X_train_pca, X_test_pca, y_train_pca, y_test_pca, label='Confusion Matrix Boosting PCA (DT depth=2)')

    print("Drawing confusion matrix for train test data")
    draw_confusion_matrix(clf_boosting, X_train_without_pca, X_test_without_pca, y_train_without_pca, y_test_without_pca, label='Confusion Matrix Boosting Train test split (DT depth=2)')





    print("Calculating the best parameters for SVM using Grid Search CV")
    param_grid = [
              {'C': [1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, param_grid)

    clf.fit(X_train, y_train)
    print("Best parameters for SVM in original training data is "+str(clf.best_params_))

    clf.fit(X_train_tsne, y_train_tsne)
    print("Best parameters for SVM in TSNE data is "+str(clf.best_params_))

    clf.fit(X_train_pca, y_train_pca)
    print("Best parameters for SVM in PCA data is "+str(clf.best_params_))

    best_params = {'C': 1, 'kernel': 'linear'}

    clf_svm = svm.SVC(kernel=best_params['kernel'], probability=True,
                                 random_state=42, C=best_params['C'])

    print("Drawing confusion matrix and ROC curves with train test split data")
    draw_confusion_matrix(clf_svm, X_train, X_test, y_train, y_test, label='Confusion Matrix SVM (Best kernel='+str(best_params['kernel'])+')')

    print("Drawing confusion matrix and ROC curves with TSNE data")
    draw_confusion_matrix(clf_svm, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, label='Confusion Matrix TSNE (Best Parmeters='+str(best_params['kernel'])+')')

    print("Drawing confusion matrix and ROC curves with PCA data")
    draw_confusion_matrix(clf_svm, X_train_pca, X_test_pca, y_train_pca, y_test_pca, label='Confusion Matrix PCA (Best Parmeters='+str(best_params['kernel'])+')')
    
    
    # create binary data for ROC curve
    y_train_binary = label_binarize(y_train, classes=np.array(list(data_with_labels.keys())))
    y_test_binary = label_binarize(y_test, classes=np.array(list(data_with_labels.keys())))

    # Do the splits again for binary conversion
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_binary, test_size=.25, random_state=42)
    X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(tsne, y_train_binary, test_size=0.25, random_state=42)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pca, y_train_binary, test_size=0.25, random_state=42)


    clf_svm = OneVsRestClassifier(svm.SVC(kernel=best_params['kernel'], probability=True,
                                 random_state=42, C=best_params['C']))

    
    (fpr_test_svm, tpr_test_svm, auc_test_svm) = get_ROC_data_SVM(clf_svm, X_train, X_test, y_train, y_test, y_test_binary, label='ROC SVM (Best kernel='+str(best_params['kernel'])+')')
    (fpr_tsne_svm, tpr_tsne_svm, auc_tsne_svm) = get_ROC_data_SVM(clf_svm, X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne, y_test_binary, label='ROC SVM with PCA (Best kernel='+str(best_params['kernel'])+')')
    (fpr_tsne_svm, tpr_tsne_svm, auc_tsne_svm) = get_ROC_data_SVM(clf_svm, X_train_pca, X_test_pca, y_train_pca, y_test_pca, y_test_binary, label='ROC SVM with PCA (Best kernel='+str(best_params['kernel'])+')')





    # print("Applying Our Decision Tree algorithm for depth=2")
    # length of training set
    length_of_data_set = len(X_train)

    # calculate weights of every example -> equal for first time
    weights = np.ones(length_of_data_set)/length_of_data_set

    decision_tree = id3(X_train, y_train, weight=weights, max_depth=2)
    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in X_test]
    tst_err = compute_error(y_test, y_pred)
    accuracy_score(y_pred, y_test)



    # Code to run the model on one classifier

    # pickle.dump(clf_svm, open('img_model.p', 'wb'))
    # model = pickle.load(open('img_model.p', 'rb'))
    # flat_data = []
    # url = input('Enter the url')
    # img = imread(url)
    # img_resize = resize(img, (100, 100, 3))
    # flat_data.append(img_resize.flatten())
    # flat_data = np.array(flat_data)
    # print(img.shape)
    # plt.imshow(img_resize)
    # y_out = model.predict(flat_data)
    # y_out = fruits[y_out[0]]
    # print(f'Predicted output : {y_out}')







    




    











