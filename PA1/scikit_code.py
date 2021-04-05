from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

def learn_decision_tree_from_scikit():
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn decision tree from scikit
    classifier = DecisionTreeClassifier()
    classifier.fit(Xtrn, ytrn)
    
    # Draw decision tree using graphviz
    fn = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
    cn = ['0', '1']
    tree.export_graphviz(classifier,out_file="tree1.dot",feature_names = fn, class_names=cn,filled = True)

    # Print the confusion matrix
    print(confusion_matrix(ytst, y_pred))