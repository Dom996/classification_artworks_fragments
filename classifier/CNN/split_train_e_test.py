import os
import sys
import time
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# ******** Dataset_F_Random_npz ************
number = 40  # Pezzi SLIC--> Dataset_F_Random
DatasetName = 'Dataset_F_Random_' + str(number) + 'pz'

# ******** Dataset_F_Aware_n_cluster ************
# number = 40  # Cluster --> Dataset_F_Aware
# DatasetName = 'Dataset_F_Aware_' + str(number) + '_cluster'

# ******** Dataset_F_RandomAware************
# n_Dataset = 3
# DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

if os.path.isdir('../../DataSet/' + DatasetName + '/Output/Classifier'):
    print()
else:
    os.mkdir('../../DataSet/' + DatasetName + '/Output/Classifier')

PATH_Dataset = '../../DataSet/' + DatasetName

PATH_All_DataSet = PATH_Dataset + '/All_DataSet/'
train_dir = PATH_Dataset + '/Train&Test/Training_Set/'
test_dir = PATH_Dataset + '/Train&Test/Test_Set/'


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


f = open('../../DataSet/' + DatasetName + '/Output/Classifier/Log Split_Train&Test_Classifier.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)


def load_imagelabels():
    x = []  # vettore delle immagini
    y = []  # vettore delle labels

    if os.path.isdir(train_dir):  # creiamo la directory Training_Set
        print()
    else:
        os.mkdir(train_dir)  # Crea nuova directory per il peridodo storico di riferimento

    if os.path.isdir(test_dir):  # creiamo la directory Training_Set
        print()
    else:
        os.mkdir(test_dir)  # Crea nuova directory per il peridodo storico di riferimento

    for artistic_period in os.listdir(PATH_Dataset + '/All_DataSet'):
        path = PATH_All_DataSet + artistic_period + '/'

        if os.path.isdir(
                train_dir + artistic_period):  # nella directory Training_Set creiamo le directory di ogni periodo artistico
            print()
        else:
            os.mkdir(train_dir + artistic_period)

        if os.path.isdir(
                test_dir + artistic_period):  # nella directory Test_Set creiamo le directory di ogni periodo artistico
            print()
        else:
            os.mkdir(test_dir + artistic_period)

        imagesList = listdir(path)  # Crea una lista di immagini

        for file in imagesList:  # per ogni immagine nella lista
            x.append(file)  # aggiungiamo ogni immagine della lista nel vettore x
            y.append(artistic_period)  # aggiungiamo ogni periodo artistico nel vettore y

    d = {'images': x, 'labels': y}  # definiamo le colonne del data frame con gli attributi images e labels
    df = pd.DataFrame(data=d)  # creiamo un data frame con 2 colonne immagini e labels
    return df


def splitTrain_Test(X_train, X_test):
    print("Save image Train: Inizio")
    count = 0
    for fragment_train in X_train:  # per ogni frammento del Training Set

        artistic_period, fragment_name = fragment_train.split(
            '_')  # es. Fragment: CUBISMO_38.6.png -----> artistic_period: CUBISMO, fragment_name: 38.6.png

        img_train = Image.open(PATH_All_DataSet + artistic_period + '/' + fragment_train)  # carichiamo le immagini
        img_train.save(train_dir + artistic_period + '/' + fragment_name)  # salviamo ogni immagine nel Training Set
        count = count + 1
    end = time.time()
    print("Training Set: " + str(count) + " immagini")
    print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min")
    print("Save image Train: Terminato\n")

    print("Save image Test: Inizio")
    count = 0
    for fragment_test in X_test:
        artistic_period, fragment_name = fragment_test.split(
            '_')  # es Fragment: CUBISMO_38.6.png -----> artistic_period: CUBISMO, fragment_name: 38.6.png

        img_test = Image.open(PATH_All_DataSet + artistic_period + '/' + fragment_test)  # carichiamo le immagini
        img_test.save(test_dir + artistic_period + '/' + fragment_name)  # salviamo ogni immagine nel Test Set
        count = count + 1
    end = time.time()
    print("Test Set: " + str(count) + " immagini")
    print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min")
    print("Save image Test: Terminato\n")


start = time.time()
print(DatasetName + ': Train&Test Split\n')

dataFrame = load_imagelabels()  # carichiamo le immagini e le labels nel data frame

# Nel seguente modo effettuiamo lo suddivisione dei frammenti in 80% Training Set e 20% Test Set con la funzione
# train_test_split
print("Train_Test_Split : Inizio")
X_train, X_test, y_train, y_test = train_test_split(dataFrame['images'], dataFrame['labels'], test_size=0.2,
                                                    shuffle=True)
end = time.time()
print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min\nTrain_Test_Split terminato\n")

splitTrain_Test(X_train,
                X_test)  # effettuiamo il salvataggio dei frammenti splittati nelle directory Training_Set e Test_Set

end = time.time()
print("\nTempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")
