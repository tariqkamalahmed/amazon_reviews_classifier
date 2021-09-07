#!/usr/bin/env python
#Tariq Ahmed 
# coding: utf-8

# In[1]:


import csv,re#, nltk                               # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.pipeline import Pipeline
#from nltk.tokenize import word_tokenize ##techniques used and did not improve or slightly reduced performance
#from nltk.stem import WordNetLemmatizer ##techniques used and did not improve or slightly reduced performance
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# In[2]:


def loadData(path, Text=None):
    with open(path, 'r',encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            (Id, Text, Rating, Verified, Category, Label) = parseReview(line)
            rawData.append((Id, Text, Rating, Verified, Category, Label))

def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    
    for (_, Text, Rating, Verified, Category, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        other_features = toFeatureVector(preProcess(Text))
        other_features.update({'Rating':Rating, 'Verified':Verified, 'Category':Category}) 
        trainData.append((other_features, Label))
    for (_, Text, Rating, Verified, Category, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        other_features = toFeatureVector(preProcess(Text))
        other_features.update({'Rating':Rating, 'Verified':Verified, 'Category':Category}) 
        testData.append((other_features, Label))
#        
     
        

# # Question 1

# In[3]:


# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):

    Id = reviewLine[0]
    Text = reviewLine[8]
    Label = reviewLine[1]
    Rating = reviewLine[2] 
    Verified = reviewLine[3]
    Category = reviewLine[4]
    
    if reviewLine[1]  == '__label1__':
        reviewLine[1] = 'fake'
    else :
        reviewLine[1] = 'real'
    

   
    return (Id, Text, Rating, Verified, Category, Label)

# The rating was chosen as maybe relation with fake reviews and the rating given
# If the reviewer was verified to help with classification and improve scores
# the category as also maybe higher fake reviews in specific categories
# Using these 3 other features, there was an improvement in performance from F Score:0.605206 to 0.731027 by itself with no other techniques added


    
# In[4]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# Input: a string of one review
def preProcess(text):
    # word tokenisation
    # separate out words and strings of punctuation into separate white spaced words
    text = re.sub(r"(\w)([.,;:!?'\"”\)])", r"\1 \2", text) #remove
    text = re.sub(r"([.,;:!?'\"“\(])(\w)", r"\1 \2", text)
    tokens = re.split(r"\s+",text)#split the regular expression by the white space
    text = re.sub(r"(\S)\1\1+",r"\1\1\1", text) #normalisation
    tokens = [t.lower() for t in tokens] 
    #tokens = word_tokenize(text)
    #tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens] 
    #tokens = [nltk.stem.SnowballStemmer('english').stem(t) for t in tokens]   

    return tokens
# Using lemmatization slightly reduced the F score from 0.604973 to 0.589483 by itself with no other techniques added
# Using word_tokenize also slightly reduced the F score from 0.604973 to 0.593786 by itself with no other techniques added
# using stemming also slightly reduced the f score from 0.604973 to 0.592549 by itself with no other techniques added

#
# # Question 2

# In[5]:


featureDict = {} # 
def toFeatureVector(words):     # returns a dictionary where the features as keys, and weights as values
    v = {}
    for w in words:
        try:
            featureDict[w] += 1
        except KeyError:            
            featureDict[w] = 1
        try:
            v[w] += (1.0/len(words))
        except KeyError:
            v[w] = (1.0/len(words))

    return v


#This increased performance by 3.5% from f score of 0.604973 to 0.637559 by itself with no other techniques added.

# In[6]:


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC())])
    return SklearnClassifier(pipeline).train(trainData)


# # Question 3

# In[7]:


def crossValidate(dataset, folds):
    shuffle(dataset)
    cv_results = []
    foldSize = int(len(dataset)/folds)
    # DESCRIBE YOUR METHOD IN WORDS
    for i in range(0,len(dataset),foldSize):
        trainFolds = dataset[i:i+foldSize]
        validationFold = dataset[:i] + dataset[i+foldSize:]
        classifier = trainClassifier(trainFolds)
        truth = [x[1] for x in validationFold]
        pred = predictLabels(validationFold,classifier)
        cv_results.append(precision_recall_fscore_support(truth, pred, average='weighted'))
    
    return cv_results


# In[8]:


# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))

def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))


# In[9]:


# MAIN


rawData = []          # the filtered data from the dataset file (should be 21000 samples)
trainData = []        # the pre-processed training data as a percentage of the total dataset (currently 80%, or 16800 samples)
testData = []         # the pre-processed test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# the output classes
fakeLabel = 'fake'
realLabel = 'real'


reviewPath = 'amazon_reviews.txt'


print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing the dataset...",sep='\n')
loadData(reviewPath) 


print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Preparing training and test data...",sep='\n')
splitData(0.8)

print("After split, %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
      "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')


crossValidate(trainData, 10) # perform 10 folds using the crossvalidate method

# using dictionary that uses the features as keys, and weights as values with the 3 additional feature increase the F score from 0.604973 from question 3 to 0.800354  


# # Evaluate on test set

# In[10]:




functions_complete = True  # set to True once you're happy with your methods for cross val
if functions_complete:
    print(testData[0])   # have a look at the first test data instance
    classifier = trainClassifier(trainData)  # train the classifier
    testTrue = [t[1] for t in testData]   # get the ground-truth labels from the data
    testPred = predictLabels(testData, classifier)  # classify the test data to get predicted labels
    finalScores = precision_recall_fscore_support(testTrue, testPred, average='weighted') # evaluate
    print("Done training!")
    print("Precision: %f\nRecall: %f\nF Score:%f" % finalScores[:3])






