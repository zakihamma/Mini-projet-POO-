#L'importation des modules utilisés 
import csv
import random
import operator
import math
#=============================================================


#=============================================================
#entrer l'emplacement de ta base de données ici (ou 
# le nom du fichier s'il était dans le même répertoire): 
filename = 'iris.data.txt'
#=============================================================


#=============================================================
#cette fonction sert séparer la base de données en partie train et partie test 
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
#=============================================================


#=============================================================
#cette fonction sert à calculer la distance euclidienne entre le 
#point à tester et les autres points
def euclideanDistance(instance1, instance2, length):
    somme = 0
    for x in range(length):
        somme = somme + (instance1[x] - instance2[x])**2
    distance = math.sqrt(somme)
    return distance
#=============================================================


#=============================================================
#cette fonction sert à déterminer les k voisins les plus 
#proches du point concerné
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#=============================================================


#=============================================================
#cette fonction sert à déterminer la classe du point concerné 
#selon la classe de la majorité des plus proches voisins 
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]  
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
#=============================================================


#=============================================================
#cette fonction sert à générer les prédictions:
def predic(trainingSet, testSet, k):
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    return predictions
#=============================================================


#=============================================================
#cette fonction sert à mesurer la précision 
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
#=============================================================


#=============================================================
#cette fonction sert à effectuer toute la procédure en 
#appelant toutes les nécessaires et affiche la précision
#à la fin
def main():
    trainingSet = []
    testSet = []
    split = 0.6
    k=5
    
    loadDataset(filename, split, trainingSet, testSet)
    predictions = predic(trainingSet, testSet, k)
    accuracy = getAccuracy(testSet, predictions)
    
    print('Accuracy: ' + repr(accuracy) + '%')
#=============================================================


#=============================================================
#cette ligne sert à exécuter le code 
main()