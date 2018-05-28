import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import re
import nltk, nltk.stem.porter
from sklearn import svm


def preProcess(email):

    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email);
    # Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email);
    # The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email);
    return email


def email2TokenList(raw_email):
    """
    Function that takes in preprocessed (simplified) email, tokenizes it,
    stems each word, and returns an (ordered) list of tokens in the e-mail
    """

    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess(raw_email)

    # Split the e-mail into individual words (tokens) (split by the delimiter ' ')
    # Splitting by many delimiters is easiest with re.split()
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)

    # Loop over each token and use a stemmer to shorten it, check if the word is in the vocab_list... if it is, store index
    tokenlist = []
    for token in tokens:

        token = re.sub('[^a-zA-Z0-9]', '', token);
        stemmed = stemmer.stem(token)
        # Throw out empty tokens
        if not len(token): continue
        # Store a list of all unique stemmed words
        tokenlist.append(stemmed)

    return tokenlist


def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open("data/vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict


def email2VocabIndices(raw_email,vocab_disct):

    # returns a list of indices corresponding to the location in vocab_dict for each stemmed word
    tokens = email2TokenList(raw_email)
    index_list = []

    for token in tokens:

        if token in vocab_disct.keys():
            index_list.append(vocab_disct[token])

    return index_list


def email2FeatureVector(raw_email, vocab_disct):

    index_list = email2VocabIndices(raw_email,vocab_disct)
    result = np.zeros((len(vocab_disct),1))

    for index in index_list:
        result[index] = 1

    return result


path_test = os.getcwd() + '/data/spamTest.mat'
path_train = os.getcwd() + '/data/spamTrain.mat'

data = loadmat(path_train)
data_test = loadmat(path_test)

X = data['X']
y = data['y']

X_test = data_test['Xtest']
y_test = data_test['ytest']

vocab_disct = getVocabDict()
raw_email = open('data/emailSample1.txt', 'r').read()
feature_vector = email2FeatureVector(raw_email,vocab_disct)

print("Length of feature vector is: " + str(len(feature_vector)))
print("Number of non-zero entries is: " + str(sum(feature_vector==1)))

# Train the SVM classifier
svc = svm.SVC(C=0.1, kernel='linear')
svc.fit(X,y.ravel())
score_train = svc.score(X, y.ravel())
score_test = svc.score(X_test,y_test.ravel())

print("Spam filter training set accuracy: " + str(score_train))
print("Spam filter test set accuracy: " + str(score_test))

# Perform prediction on example email
raw_email = open('data/emailSample2.txt', 'r').read()
feature_vector = email2FeatureVector(raw_email,vocab_disct)
prediction = svc.predict(feature_vector.T)
print(prediction)

coeff = svc.coef_
coeff_sorted = np.argsort(coeff)[0,:]

vocab_dict_flipped = getVocabDict(reverse=True)
predictors_pos = []
predictors_neg = []
for x in coeff_sorted[0:15]:

    predictors_pos.append(vocab_dict_flipped[x])


for x in coeff_sorted[len(coeff_sorted)-15:len(coeff_sorted)]:

    predictors_neg.append(vocab_dict_flipped[x])


print("The 15 most predictive words for spam are: \n " + str(predictors_pos))
print("The 15 least predictive words for spam are: \n " + str(predictors_neg))








