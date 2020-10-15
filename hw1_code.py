'''
Jinansh Shah
25/01/19
v1

The following uses scikit-learn decision tree classifier to classify real vs. fake
news headlines.

'''


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd
import graphviz 


#Global Variables:
# vec stores the initialized CountVectorizer object - it is used in multiple functions
vec = CountVectorizer()

#The following function loads the fake and real headlines and returns a transformed training set (70%) and validation set (15%). 
def load_data():
    fake, real = "", ""
    fake += open("clean_fake.txt", "r").read()
    fake_list = fake.split('\n')  
    real += open("clean_real.txt", "r").read()
    real_list = real.split('\n')
    

    df_fake = pd.DataFrame({'Headline': fake_list, 'Label': 'FAKE'})
    df_real = pd.DataFrame({'Headline': real_list, 'Label': 'REAL'})
    df_all = pd.concat([df_fake, df_real])
    X = vec.fit_transform(fake_list+real_list)
    
    X_train, X_test, y_train, y_test  = train_test_split(X, df_all['Label'], test_size=0.3, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    
    return X_train, y_train, X_val, y_val


#The builds the DecisionTreeClassifier and trains it using the training set from load_data() and prints the accuracy of trees of various depths (1, 2, 4, 9, 16, 40, 80, 120) and criteria (gini vs entropy) using the validation set. 
def select_model():
    X_train, y_train, X_val, y_val = load_data()
    x = X_val.toarray()
    y = y_val.values
    L= []
    for criteria in ['gini', 'entropy']:
        for max_depth in [1, 2, 4, 9, 16, 40, 80, 120]:
            print('Split Criteria: ', criteria, '; Max Depth: ', max_depth)
            clf = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criteria)
            clf.fit(X_train, y_train)
            
            # validation code which does the same thing as clf.score(X_val, y_val)
            correct = 0
            for i in range (0, len(x)):
                if clf.predict([x[i]]) == y[i]:
                    correct = correct + 1
            print('Accuracy: ', correct/len(x), '\n')
            L.append(correct/len(x))
            dot_data=tree.export_graphviz(clf, feature_names=vec.get_feature_names(),  out_file='tree.dot') 
    print('Best result: ', max(L))

#determines and prints the information gain at keyword xi
def compute_information_gain(Y, xi):
    X_train, y_train, X_val, y_val = load_data()
    res = dict(zip(vec.get_feature_names(),mutual_info_classif(X_train, y_train, discrete_features=True)))
    print ('IG for the word [', xi, ']:', res[xi])
   
    


#calls select_model and compute_information_gain
if __name__ == '__main__':
    print('-------select_model()-----------')
    select_model()
    print('-------information_gain(Y, xi)---------')
    compute_information_gain('FAKE', 'donald')
    compute_information_gain('FAKE', 'trumps')
    compute_information_gain('FAKE', 'hillary')
    compute_information_gain('FAKE', 'fame')
    compute_information_gain('FAKE', 'and')
    compute_information_gain('FAKE', 'as')
