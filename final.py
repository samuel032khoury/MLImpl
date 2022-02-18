import sys
import math
import random
import numpy as np
import pandas as pd
from numpy import linalg as la

### I. Common Helpers

def quantCol(dataFrame, column):
    '''
    Collects all the possible values of the inspecting column of the provided dataFrame,then assign
    an integer tag to each of them, finally returns a dictionary that maps the original value to
    that numerical tag.

    Parameters
    ----------
        dataFrame : pandas.core.frame.DataFrame
            a data frame that supplies data
        column : str
            the name of the inspecting column whose values are not numbers

    Returns
    -------
        out : {obj : int}
            a dictionary that maps the original value to the assigned numerical tag.

    Side Effect
    -----------
        Replace the original value of the inspecting column with the corresponding integer tag.
    '''
    optionList = dataFrame[column].unique()
    optionList.sort()
    optionNominal = {value: (index + 1) for index, value in enumerate(optionList)}
    dataFrame[column].replace(optionNominal, inplace=True)
    return optionNominal

def compSimilarity(expect, actual):
    '''
    Compute the similarity of two provided lists, by the percentage of two lists have the same 
    elements at each index.

    Parameters
    ----------
        expect : [int]
            the expected list of integers
        actual : [int]
            the actual (comparing) list of integers

    Returns
    -------
        out : float
            the similarity of two provided lists, ranged from 0 to 1, with 1 representing the two 
            lists have exactly the same contents
    Raises
    ------
        Exception
            if the two lists don't share the same size
    '''
    if not (len(expect) == len(actual)):
        raise Exception("Unable to compare two list because they don't have the same size!")
    if (expect == actual):
        return 1
    accuracy = 0
    step = 1 / len(expect)
    for i in range(len(expect)):
        if(expect[i] == actual[i]):
            accuracy += step
    return accuracy

def varifyTraining(label, trainingFile):
    '''
    varify the training file is meaningful for the program, i.e. the label column exists.

    Parameters
    ----------
        label : str
            the name of the label column for prediction
        trainingFile : str
            the file path of the training file as a string

    Returns
    -------
        out : pandas.core.frame.DataFrame
            a dataframe of the training file with all invalid entries removed

    Raises
    ------
        AttributeError
            if there is not a label column in the training file
    '''
    trainingDF = pd.read_csv(trainingFile).dropna()  # read a file as a dataframe
    
    if not (label in trainingDF):
        raise AttributeError("No such attribute {label} for classification!".format(label = label))
    return trainingDF

def varifyTargeting(label, targetingFile):
    '''
    varify the targeting file is meaningful for the program, i.e. the label column (if exists) is
    empty.

    Parameters
    ----------
        label : str
            the name of the label column for prediction
        targetingFile : str
            the file path of the targeting file as a string

    Returns
    -------
        out : pandas.core.frame.DataFrame
            a dataframe of the targeting file with all invalid entries removed

    Raises
    ------
        AttributeError
            if there is a label column in the targeting file, but the column is not empty
    '''
    targetingDF = pd.read_csv(targetingFile)  # read the targeting file as a dataframe

    if label in targetingDF:
        if not targetingDF[label].isna().all():
            raise AttributeError("Column for the predicting attribute has to be empty!")
        targetingDF = targetingDF.drop(columns=[label]).dropna()

    return targetingDF

def checkValidAttr(dataFrame, selectedAttr):
    '''
    varify that all the selected attributes to inspect exists in the provided dataframe

    Parameters
    ----------
        dataFrame : pandas.core.frame.DataFrame
            the data frame to inspect
        selectedAttr : [str]
            the list of string representing selected attributes(columns) for prediction

    Raises
    ------
        AttributeError
            if any attribute in the list does not exist in the dataframe
    '''
    for attr in selectedAttr:
        if not attr in dataFrame:
            raise AttributeError("No attribute {attr} found".format(attr = attr))

def genTargetAttrList(row, entry, targetingAttr, attributeNominalMaps):
    '''
    Given a row of data, standardized the entry by the integer tag, and generate an attribute list
    with the selected attributes.

    Parameters
    ----------
        row : int
            the row of the data currently processing
        entry : pandas.core.frame.Series
            a series of data for the current row
        targetingAttr : [str]
            the selected attributes for targeting data
        attributeNominalMaps : {str : {str : int}}
            the map contains the mapping from the string entry to integer labelsfor all string-based
            features

    Returns
    -------
        out : [int]
            a list of integer represents the values of targeting features
    Raises
    ------
        AttributeError
            if the targeting file contains meaningless data (i.e. unrecorded lables/string entry for
            a numeric feature column)

    '''
    entryMap = entry.to_dict()
    targetAttrList = []
    for currAttr in targetingAttr:
        if currAttr in attributeNominalMaps:
            if entryMap[currAttr] in attributeNominalMaps[currAttr]:
                targetAttrList.append(attributeNominalMaps[currAttr][entryMap[currAttr]])
            else:
                raise AttributeError(''.join(("The provided attribute value ",
                    entryMap[currAttr], " of entry ", str(row), " for ", currAttr, " is not a ",
                    "recorded label yet, a prediction therefore cannot be performed!")))
        else:
            if isinstance(entryMap[currAttr], str):
                raise AttributeError("Value for " + currAttr + " of entry " + str(row) \
                    + " should be a number!")
            targetAttrList.append(entryMap[currAttr])
    return targetAttrList

### II. KNN Section

###### II.1 KNN Core

def classify(classified, unclassified, k):
    '''
    classify a list of objects using knn algorithm, based on the classified data and the provided k.

    Parameters
    ----------
        classified : [(int, numpy.array)]
            a list of tuples representing a list of classified objects, the first item of a tuple
            represents the integer tag of the object, the second item represents its feature vector

        unclassified : [numpy.array]
            a list of vectors representing unclassifed objects
        k : int
            the number of the nearest neighbors to find

    Returns
    -------
        out : [ [(int, numpy.array)] ]
            generate a list contaning lists of candidates (k-neighbors) for every unclassified item.
             The count of the candidates depends on the k.
    '''
    result = []
    for target in unclassified:
        neighbors = [(None, float('inf'))] * k
        for classifiedNode in classified:
            maxValIndex = neighbors.index(max(neighbors, key = lambda neighbor : neighbor[1]))
            currDistance = distance(target, classifiedNode[1])
            if(currDistance < neighbors[maxValIndex][1]):
                neighbors[maxValIndex] = (classifiedNode[0], currDistance)
        result.append(neighbors)
    return result

def findKAccuracyPair(training, testing, upperLimit, step):
    '''
    to find a k that maximize the prediction accuracy for the testing data, provided the training 
    reference.

    Parameters
    ----------
        training : [(int, numpy.array)]
            a list of tuples representing a list of training objects, the first item of a tuple
            represents the integer tag of the object, the second item represents its feature vector
        testing : [(int, numpy.array)]
            a list of tuples representing a list of testing objects, the first item of a tuple
            represents the integer tag of the object, the second item represents its feature vector
        upperLimit : int
            the upper limit for k
        step : int
            the increase step for k-finding process, the step depends on the possible values for the
            labels, to avoid tie situations

    Returns
    -------
        out : (int, float)
            the k-accuracy pair with the k value maximize the prediction accuracy for the testing
            data
    '''
    kAccuracyPair = (-1, -1)
    expectLabels = getLabels(testing)
    for i in range(1,upperLimit + 1, step):
        actualLabels = list(map(lambda knn : mostFreq(getLabels(knn)),
            classify(training, getVectors(testing), i)))
        currAccuracy = compSimilarity(expectLabels, actualLabels)
        if (currAccuracy > kAccuracyPair[1]):
            kAccuracyPair = (i, currAccuracy)
        if currAccuracy == 1:
            break
    return kAccuracyPair

def knnTrainRun(label, trainingFile, targetingFile, selectedAttr):
    '''
    Predict labels for the targeting file by applying KNN algorithm on the provided targeting file. 

    Parameters
    ----------
        label : str
            the label which is going to predict
        trainingFile: str
            the file path of the traning file which is used to train KNN algorithm
        targetingFile : str
            the file path of the targeting file for prediction
        selectedAttr : [str]
            valid features in the provided dataset which uses to train the machine

    Returns
    -------
        out : (pandas.core.frame.DataFrame, str)
            the processed targeting dataframe to be saved in the machine, with a result string 
            containing the most effective k and the accuracy of the testing prediction using the
            knn algorithm

    Raises
    ------
        AttributeError
            if any selected attribute doesn't appear in the training file or the targeting file
    '''
    ### Verify section
    trainingDF = varifyTraining(label, trainingFile)
    trainingAttr = ([col for col in list(trainingDF.columns) if not col.startswith("Unnamed")]
        if len(selectedAttr) == 0 else selectedAttr + [label])

    try:
        checkValidAttr(trainingDF, trainingAttr)
    except AttributeError as e:
        raise AttributeError(str(e) + " in training file!")
    
    targetingDF = varifyTargeting(label, targetingFile)
    targetingAttr = [attr for attr in trainingAttr if attr != label]

    try:
        checkValidAttr(targetingDF, targetingAttr)  
    except AttributeError as e:
        raise AttributeError(str(e) + " in targeting file!")
    ### End verify section

    ### String labeling section
    attributeNominalMaps = {}
    for attribute in trainingAttr:
        if(trainingDF[attribute].dtype == "object" or attribute == label):
            attributeNominalMaps[attribute] = (quantCol(trainingDF, attribute))
    labelNominal = attributeNominalMaps[label]
    labelDict = dict([reversed(i) for i in labelNominal.items()])
    ### End string labeling section

    ### K-finding section
    trainingAttrTable = trainingDF[trainingAttr].astype("object")
    trainingData = genLabeledVectorList(list(trainingDF[label]), trainingAttrTable[targetingAttr])
    dataSize = len(trainingData)
    splitIndex = np.round(dataSize * 0.75).astype('int')
    random.Random(dataSize - 1).shuffle(trainingData)
    trainingSub = (trainingData[:splitIndex])
    testingSub = (trainingData[splitIndex:])

    kUpperLimit = min(10,np.ceil((np.sqrt(len(trainingSub))) / 2).astype('int'))
    kAccuracyPair = findKAccuracyPair(trainingSub, testingSub, kUpperLimit, len(labelDict))
    k = kAccuracyPair[0]
    confidence = kAccuracyPair[1]
    ### End k-finding section

    ### Prediction Core
    targetVectors = genTargetVectors(targetingDF, targetingAttr, attributeNominalMaps)
    resultList = list(map(lambda knn : labelDict[mostFreq(getLabels(knn))],
        classify(trainingSub, targetVectors, k)))
    ### Prediction Core

    targetingDF[label] = resultList
    resultString = "KNN prediction result has been saved to {filePath}"\
    + " (with k = " + str(k) + " that correctly predict " + str(round((confidence * 100), 2)) \
    + "% of the testing data)"
    return (targetingDF[targetingAttr + [label]], resultString)

###### II.2 KNN Helpers

def distance(vec1, vec2):
    '''
    Calculate the Euclidean distance between two vectors.

    Parameters
    ----------
        vec1 : numpy.array
            the first vector
        vec2 : numpy.array
            the second vector

    Returns
    -------
        out : float
            the Euclidean distance between two vectors

    Raises
    ------
        Exception
            when two vectors are not in the same space (don't have the same length)
    '''
    if not len(vec1) == len (vec2):
        raise Exception(
            "Cannot calculate the distance for two vectors that are in different spaces!")
    return la.norm(vec2 - vec1)

def mostFreq(potentialLabelList):
    '''
    Find the most frequent item in the provided non-empty list.
    
    If the frequencies of two (or more) items tie, return the first found one.

    Parameters
    ----------
        potentialLabelList : [int]
            a non-empty list of integers

    Returns
    -------
        out : int
            the most frequent item in the provided list.

    Raises
    ------
        Exception
            if the  provided list is empty.
    '''
    if len(potentialLabelList) == 0:
        raise Exception("The provided list is empty!")
    return max(set(potentialLabelList), key = potentialLabelList.count)

def genLabeledVectorList(labelList, attrTable):
    '''
    Provided an aligned label (integer) list and the features of that label, generate a tuple
    that pairs up an integer label and the corresponding row as a vector in the attrTable.

    Parameters
    ----------
        labelList : [int]
            a list of integer representing labels
        attrTable : pandas.core.frame.DataFrame
            the dataframe with its row aligning the order of the labelList.
            It contains all the related feature of an object.

    Returns
    -------
        out : [(int, numpy.array)]
            a tuple that pairs up the label and the features as a vector
    '''
    featureValueList = [];
    for i in range(len(attrTable)):
        uniqueTuple = (labelList[i], attrTable.iloc[i].values)
        featureValueList.append(uniqueTuple)
    return featureValueList

def getLabels(labelVectorPairList):
    '''
    Get the labels of a label-vector paired list

    Parameters
    ----------
        labelVectorPairList : [(int, numpy.array)]
            the label-vector paired list to break down

    Returns
    -------
        out : [int]
            the list of labels from the provided label-vector paired list
    '''
    return list(map(lambda tupleEle: tupleEle[0], labelVectorPairList))

def getVectors(labelVectorPairList):
    '''
    Get the vectors of a label-vector paired list

    Parameters
    ----------
        labelVectorPairList : [(int, numpy.array)]
            the label-vector paired list to break down

    Returns
    -------
        out : [numpy.array]
            the list of vectors from the provided label-vector paired list
    '''
    return list(map(lambda tupleEle: tupleEle[1], labelVectorPairList))

def genTargetVectors(targetingDF, targetingAttr, attributeNominalMaps):
    '''
    convert the original target data frame to the standardized (integer labeled) vector list, each
    vector is only consists of the standardized value of the selected attributes.


    Parameters
    ----------
        targetingDF : pandas.core.frame.DataFrame
            the original dataframe with the original unstandardized entry
        targetingAttr : [str]
            the selected attributes for targeting data
        attributeNominalMaps : {str : {str : int}}
            the map contains the mapping from the string entry to integer labelsfor all string-based
            features

    Returns
    -------
        out : [numpy.array]
            a list of standardized vector, with each vector is only consists of the standardized 
            value of the selected attributes.
    '''
    targetVectors = []
    for row, entry in targetingDF[targetingAttr].iterrows():
        targetAttrList = genTargetAttrList(row, entry, targetingAttr, attributeNominalMaps)
        targetVector = np.array(targetAttrList)
        targetVectors.append(targetVector)
    return targetVectors

### III. GNB Section

###### III.1 GNB Core

def gnb(x, y_mean, y_std):
    '''
    Calculate the Gaussian probability distribution for x.

    Parameters
    ----------
        x : float
            the calculating value
        y_mean : float
            the mean value for the kind
        y_std : float
            the standard deviation value for the kind

    Returns
    -------
        out : float
            the Gaussian probability distribution.
    '''
    std2 = y_std ** 2
    sqrt = math.sqrt(2 * math.pi * std2)
    exp = math.exp(-((x - y_mean)**2) / (2 * std2))
    prob = (1/sqrt) * exp
    return prob

def gnbPredict(summaries, targets):
    '''
    Predicting the label for entries in the targets

    Parameters
    ----------
        summaries : {int : ([float], [float], int)}
            the key of the dictionary representing each distinct label
            the value of the dictinary is a tuple consists of
                0. list of mean values for each features
                1. list of std values for each features
                2. the size of the entries that belong to this label

        targets : pandas.core.frame.DataFrame
            the dataframe for label prediction

    Returns
    -------
        out : list
            the predicted label by the Gaussian Naive Bayes algorithm.
    '''
    sampleSize = sum([summaries[label][2] for label in summaries])
    listOfProb = []
    for row, entry in targets.iterrows():
        probabilities = {}
        for label, summary in summaries.items():
            probabilities[label] = summaries[label][2] / sampleSize
            for i in range(len(entry.values)):
                value = entry.values[i]
                probabilities[label] *= gnb(value, summary[0][i], summary[1][i])   
        listOfProb.append(probabilities)
  
    def normalize(dict):
        summation = sum(dict.values())
        for key, value in dict.items():
            dict[key] = round(value /  summation, 2)
        return dict

    # map from a list of dictionaries, with the key representing integer labels and value
    # representing the probability, to a list of integer labels whose probability is the highest
    return list(map(lambda currDic : max(currDic, key =currDic.get),(map(normalize, listOfProb))))

def gnbTrainRun(label, trainingFile, targetingFile, selectedAttr):
    '''
    Predict labels for the targeting file by applying Gaussian Naive Bayes algorithm on the provided 
    targeting file. 

    Parameters
    ----------
        label : str
            the label which is going to predict
        trainingFile: str
            the file path of the traning file which is used to train Gaussian Naive Baye algorithm
        targetingFile : str
            the file path of the targeting file for prediction
        selectedAttr : [str]
            valid features in the provided dataset which uses to train the machine

    Returns
    -------
        out : (pandas.core.frame.DataFrame, str)
            the processed targeting dataframe to be saved in the machine, with a result string 
            containing the the accuracy of the testing prediction using the gnb algorithm

    Raises
    ------
        AttributeError
            if any selected attribute doesn't appear in the training file or the targeting file or
            the provided training data is not normally distributed.
    '''

    ### Verify section
    trainingDF = varifyTraining(label, trainingFile)
    trainingAttr = ([col for col in list(trainingDF.columns) if not col.startswith("Unnamed")]
        if len(selectedAttr) == 0 else selectedAttr + [label])

    try:
        checkValidAttr(trainingDF, trainingAttr)
    except AttributeError as e:
        raise AttributeError(str(e) + " in training file!")
    
    targetingDF = varifyTargeting(label, targetingFile)
    targetingAttr = [attr for attr in trainingAttr if attr != label]

    try:
        checkValidAttr(targetingDF, targetingAttr)  
    except AttributeError as e:
        raise AttributeError(str(e) + " in targeting file!")
    ### End verify section

    ### String labeling section
    attributeNominalMaps = {}
    for attribute in trainingAttr:
        if(trainingDF[attribute].dtype == "object" or attribute == label):
            attributeNominalMaps[attribute] = (quantCol(trainingDF, attribute))
    labelNominal = attributeNominalMaps[label]
    labelDict = dict([reversed(i) for i in labelNominal.items()])
    ### End string labeling section

    ### Data statistics summaries generation section
    trainingAttrTable = trainingDF[trainingAttr].astype("object")
    trainingSub = trainingAttrTable.sample(frac=0.75,random_state=200)
    testingSub = trainingAttrTable.drop(trainingSub.index)
    separated = {}

    for row, entry in trainingSub.iterrows():
        if not entry[label] in separated:
            separated[entry[label]] = []
        separated[entry[label]].append((entry[targetingAttr].values))

    summaries = {}
    for key in separated:
        labelStats = pd.DataFrame(separated[key])
        mean = labelStats.mean().values
        stdev = labelStats.std().values
        if 0 in stdev or np.isnan(stdev).any():
            raise AttributeError("The provided training data is not normally distributed,"\
            +" GNB cannot be performed!")
        sampleSize = len(labelStats)
        summaries[key] = (mean, stdev, sampleSize)
    ### End data statistics summaries generation section
        
    expect = list(testingSub[label].values)
    actual = gnbPredict(summaries, testingSub[targetingAttr])
    accuracy = compSimilarity(expect, actual)

    ### Prediction Core
    targetDataFrame = genTargetDataFrame(targetingDF, targetingAttr, attributeNominalMaps)
    targetingDF[label] = list(map(lambda x : labelDict[x], gnbPredict(summaries,targetDataFrame)))
    ### Prediction Core

    resultString = "GNB prediction result has been saved to {filePath}"\
        + " (with an accuracy that correctly predict " + str(round((accuracy * 100), 2)) \
        + "% of the testing data)"
    return (targetingDF[trainingAttr], resultString)

###### III.2 GNB Helpers

def genTargetDataFrame(targetingDF, targetingAttr, attributeNominalMaps):
    '''
    convert the original target data frame to the standardized (integer labeled) dataframe, and only
    with selected attribute columns.

    Parameters
    ----------
        targetingDF : pandas.core.frame.DataFrame
            the original dataframe with the original unstandardized entry
        targetingAttr : [str]
            the selected attributes for targeting data
        attributeNominalMaps : {str : {str : int}}
            the map contains the mapping from the string entry to integer labelsfor all string-based
            features

    Returns
    -------
        out : pandas.core.frame.DataFrame
            a standardized dataframe, with all entry as integers, and only with selected attribute
            columns
    '''
    targetDataFrame = pd.DataFrame(columns=targetingAttr)
    for row, entry in targetingDF[targetingAttr].iterrows():
        targetAttrList = genTargetAttrList(row, entry, targetingAttr, attributeNominalMaps)
        targetDataFrame.loc[row] = targetAttrList
    return targetDataFrame

### IV. Main Section

def runSample():
    main(["main", "Drug",
     "./training_data/drug200_training.csv", "./target_data/drug200_target.csv"])

    main(["main", "fruit_name",
     "./training_data/fruit_data_with_colours_training.csv",
    "./target_data/fruit_data_with_colours_target.csv", "mass", "width", "height", "color_score"])

    main(["main", "HeartDisease",
     "./training_data/heart_training.csv", "./target_data/heart_target.csv"])

    main(["main", "Species",
     "./training_data/Iris_training.csv", "./target_data/Iris_target.csv"])

    main(["main", "class",
     "./training_data/mushrooms_training.csv", "./target_data/mushrooms_target.csv"])

    main(["main", "Walc",
     "./training_data/student_training.csv", "./target_data/student_target.csv",
     "sex", "age", "famsize", "Pstatus", "Medu", "Fedu", "studytime", "activities", "freetime",
     "health", "absences"])

def main(argv):
    '''
    Run the entire program by using the command lines. 

    Parameters
    ----------
        argv[1] : string
            the label which are going to predict
        argv[2] : file path of the csv training file
            the traning file which is used to train the data by kNN and Gaussian Naive Bayes
        argv[3] : file path of the csv targeting file
            the targeting file for prediction
        argv[4:] : (optional) string...
            valid features in the provided dataset which uses to train the machine

    Effects
    -------
        write two result csv file with the predicted label by the kNN and Gaussian Naive Bayes
        algorithm into the machine.
    '''
    if len(argv) == 2 and argv[1] == "--sample":
        runSample()
        return

    if len(argv) < 4:
        print("ERROR: Insufficient arguments, please specify predicting label, the training file,",
            "and the targeting file, or input --sample as the only argument to",
            "run the sample dataset!")
        return
    label = argv[1]
    trainingFile = argv[2]
    targetingFile = argv[3]
    selectedAttr = argv[4:]
    

    if label in selectedAttr:
        print("ERROR: A feature cannot be the label and an attribute at the same time!")
        return

    if not trainingFile[-4:] == ".csv":
        print("ERROR: Please use a csv format file for training!")
        return
    if not targetingFile[-4:] == ".csv":
        print("ERROR: Please use a csv file for importing target objects!")
        return
    try:
        pd.read_csv(trainingFile) ## try read the training file as a dataframe
    except FileNotFoundError:
        print("ERROR: Unable to find the training file:", trainingFile + ",",
         "please check the name or the path of the file is accurate and try again!")
        return
    try:
        pd.read_csv(targetingFile) ## try read the targeting file as a dataframe
    except FileNotFoundError:
        print("ERROR: Unable to find the targeting file:", targetingFile + ",",
         "please check the name or the path of the file is accurate and try again!")
        return

    try:
        
        knnResult = knnTrainRun(label, trainingFile, targetingFile, selectedAttr)
        import os
        if not os.path.exists("./out"):
            os.makedirs("./out")
        knnSaveLoc = "./out/knn_" + os.path.relpath(
            targetingFile,os.path.dirname(targetingFile))[:-4] + "_result.csv"
        knnResult[0].to_csv(knnSaveLoc, index=False)
        print(knnResult[1].format(filePath = knnSaveLoc))

        gnbResult = gnbTrainRun(label, trainingFile, targetingFile, selectedAttr)
        gnbSaveLoc = "./out/gnb_" + os.path.relpath(
            targetingFile,os.path.dirname(targetingFile))[:-4] + "_result.csv"
        gnbResult[0].to_csv(gnbSaveLoc, index=False)
        print(gnbResult[1].format(filePath = gnbSaveLoc))
    except AttributeError as e:
        print("ERROR: " + str(e))
    except KeyboardInterrupt:
        print(" Program was quit")
    
if __name__ == '__main__':
    main(["main", "--sample"])
    main(sys.argv)
