import sys
import sys
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
import pickle as pk
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


# ******** Dataset_F_RandomAware************
n_Dataset = 4
DatasetName = 'Dataset_F_RandomAware_'+str(n_Dataset)

# ******** Dataset_F_Random_20pz ************
# number = 20 #Pezzi SLIC--> Dataset_F_Random
# DatasetName = 'Dataset_F_Random_' + str(number) + 'pz'

# ******** Dataset_F_Aware_10_cluster ************
# number = 20  # Cluster --> Dataset_F_Aware
# DatasetName = 'Dataset_F_Aware_' + str(number) + '_cluster'

if os.path.isdir('../../DataSet/' + DatasetName + '/Output/Random_Forest'):
    print("Loading..")
else:
    os.mkdir('../../DataSet/' + DatasetName + '/Output/Random_Forest')

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

f = open('../../DataSet/'+DatasetName+ '/Output/Random_Forest/Log RandomForest_Training.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

main_directory_dataset = '../../DataSet/' + DatasetName + '/Train&Test/'
train_path = main_directory_dataset + 'Train'

# numero intervalli di classe dell'istogramma
bins = 8

# dimensioni per il ridimensionamento dei frammenti
fixed_size = tuple((700, 700))

# estrazione delle labels dei dati di training
train_labels = os.listdir(train_path)

# Ordinamento delle labels di training
train_labels.sort()

# creazione di liste vuote per mantenere i feature vectors e le labels
global_features = []
labels = []


# features descriptor -1:  Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertiamo in scala di grigi
    feature = cv2.HuMoments(cv2.moments(image)).flatten()  # Calcoliamo i momenti di Hu

    return feature


# feature-descriptor -2: Haralick Texture
def fd_haralick(image):
    # Convertiamo l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcolo texture di Haralick
    haralic = mahotas.features.haralick(gray).mean(axis=0)

    return haralic


# feature-description -3: Color Histogram
def fd_histogram(image, mask=None):
    # conversione dell'immagine in spazio di colori HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Computa l'istogramma
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    # Normalizzazione dell'istogramma
    cv2.normalize(hist, hist)

    # Ritorna l'istogramma come vettore monodimensionale
    return hist.flatten()

start = time.time()
print('Train Random_Forest: '+ DatasetName)

for training_name in train_labels:
    dir = os.path.join(train_path, training_name)

    # Label di training corrente
    current_label = training_name

    # iterazione delle immagini in ogni sottocartella
    for file in os.listdir(dir):

        file = dir + "/" + os.fsdecode(file)

        #lettura delle immagini e ridimensionamento in una dimensione fissa
        image = cv2.imread(file)

        if image is not None:  # Calcolo delle features per ogni immagine
            image = cv2.resize(image, fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            fv_histogram = fd_histogram(image)

        # concatenazione delle feature globali in uno Stack
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])  #

        # Aggiornamento delle liste delle label e dei features vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] Cartella Processata: {}".format(current_label))

    # Codifica delle etichette di destinazione con un valore compreso tra 0 e n_classes-1.
    targetNames = np.unique(labels)
    le = LabelEncoder()  # Nel nostro caso da 0 a 10, per 11 Periodi Artistici
    target = le.fit_transform(labels)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

global_features = np.array(rescaled_features)
global_labels = np.array(target)

print("[STATUS] Fine dell'estrazione delle Features del Training Set")

# n_estimators = [400, 600, 800]
# max_depth = [16, 32, 64]
# tuned_parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
#
# clf = GridSearchCV(ensemble.RandomForestRegressor(), tuned_parameters, n_jobs=-1, verbose=10)
# clf.fit(global_features, global_labels)
# print(clf.best_estimator_)
#
# # create the model - Random Forests
# clf_Random = RandomForestClassifier(n_estimators=clf.best_params_['n_estimators'], max_depth=clf.best_params_['max_depth'])

clf_Random = RandomForestClassifier(n_estimators=600, max_depth=32)
# fit the training data to the model
clf_Random.fit(global_features, global_labels)
with open('../../DataSet/' + DatasetName + '/Output/Random_Forest/model_Random_Forest.h5', 'wb') as f:
    pk.dump(clf_Random, f)
print('Salvataggio Modello')

end = time.time()
print("\nTempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")