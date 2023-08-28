import sys
import sys
import time
import numpy as np
import pickle as pk

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import mahotas
import matplotlib.pyplot as plt

n_Dataset = 1
# ******** Dataset_F_RandomAware ************
DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

# ******** Dataset_F_Random_20pz ************
# number = 20  # Pezzi SLIC--> Dataset_F_Random
# DatasetName = 'Dataset_F_Random_' + str(number) + 'pz'

# ******** Dataset_F_Aware_10_cluster ************
number = 10  # Cluster --> Dataset_F_Aware
DatasetName = 'Dataset_F_Aware_' + str(number) + '_cluster'


path_classifier = ['../../DataSet/' + DatasetName + '/Output/Random_Forest',
                   '../../DataSet/' + DatasetName + '/Output/SVM']

for path_dir in path_classifier:
    if os.path.isdir(path_dir):
        print("Loading..")
    else:
        os.mkdir(path_dir)


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


f = open('../../DataSet/' + DatasetName + '/Output/SVM/Log SVM_Test.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

main_directory_dataset = '../../DataSet/' + DatasetName + '/Train&Test/'

test_path = main_directory_dataset + 'Test_Set'

# numero intervalli di classe dell'istogramma
bins = 8

# dimensioni per il ridimensionamento dei frammenti
fixed_size = tuple((700, 700))

# get the training data labels
test_labels = os.listdir(test_path)
test_labels.sort()

target_names = ['EGIZIA', 'ETRUSCA', 'GRECA', 'PREISTORICA', 'ROMANA', 'BIZANTINA', 'CUBISMO', 'IMPRESSIONISMO',
                'RINASCIMENTO', 'GOTICO', 'SURREALISMO']
global_features_test = []
labels_test = []


# features description -1:  Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
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


def loadModel():
    with open('../../DataSet/' + DatasetName + '/Output/SVM/model_SVM.h5', 'rb') as f:
        clf = pk.load(f)
    print('Lettura Modello Completata')
    return clf


def predict(clf):
    preds = clf.predict(global_features_test)  # Effettuiamo le predizioni

    # Stampiamo il report sulla predizione effettuata
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(global_labels_test, preds, target_names=target_names)))
    plt.rcParams.update({'font.size': 7})

    # Calcoliamo la matrice di confusione
    disp = metrics.plot_confusion_matrix(clf, global_features_test, global_labels_test, display_labels=target_names,
                                         xticks_rotation=25)

    disp.figure_.suptitle("Confusion Matrix")  # Impostiamo il titolo della matrice di confusione

    # Salviamo la matrice di confusione
    plt.savefig('../../DataSet/' + DatasetName + '/Output/SVM/Confusion Matrix_SVM.png')


start = time.time()

print('Test SVM: ' + DatasetName)

for test_name in test_labels:
    # join the training data path and each species training folder
    dir = os.path.join(test_path, test_name)

    # get the current training label
    current_label = test_name

    # loop over the images in each sub-folder
    for file in os.listdir(dir):
        file = dir + "/" + os.fsdecode(file)

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)

        if image is not None:
            image = cv2.resize(image, fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            fv_histogram = fd_histogram(image)

        # Concatenate global features
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels_test.append(current_label)
        global_features_test.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

# encode the target labels
targetNames_test = np.unique(labels_test)
le = LabelEncoder()
target_test = le.fit_transform(labels_test)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features_test = scaler.fit_transform(global_features_test)

print("[STATUS] end of features extraction of test set...")

global_features_test = np.array(rescaled_features_test)
global_labels_test = np.array(target_test)

clf = loadModel()
predict(clf)

end = time.time()
print("\nTempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")
