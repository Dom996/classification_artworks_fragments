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
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


f = open('../../DataSet/' + DatasetName + '/Output/Random_Forest/Log RandomForest_Test.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

main_directory_dataset = '../../DataSet/' + DatasetName + '/Train&Test/'

test_path = main_directory_dataset + 'Test'

# numero intervalli di classe dell'istogramma
bins = 8

# dimensioni per il ridimensionamento dei frammenti
fixed_size = tuple((700, 700))

test_labels = os.listdir(test_path)
test_labels.sort()

target_names = ['EGIZIA', 'ETRUSCA', 'GRECA', 'PREISTORICA', 'ROMANA', 'BIZANTINA', 'CUBISMO', 'IMPRESSIONISMO',
                'RINASCIMENTO', 'GOTICO', 'SURREALISMO']
global_features_test = []
labels_test = []


# features descriptor -1:  Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertiamo l'immagine in scala di grigi

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


def loadModel():
    # Carichiamo il modello
    with open('../../DataSet/' + DatasetName + '/Output/Random_Forest/model_Random_Forest.h5', 'rb') as f:
        clf = pk.load(f)

    print('Lettura Modello Completata')

    return clf


# funzione per disegnare la matrice di confusione
def plot_confusion_matrix(cm, classes, normalize=False, title='Matrice di confusione', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # boh
    plt.title(title)  # Impostiamo il titolo del grafico
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=20, fontsize=5)  # Impostiamo le label sull'asse delle ascisse
    plt.yticks(tick_marks, classes, fontsize=7)  # Impostiamo le label sull'asse delle ordinate

    # Controlliamo se la matrice di confusione è normalizzata
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    plt.ylabel('Label Effettiva')  # Impostiamo il nome della label sull'asse delle ordinate
    plt.xlabel('Label Predetta')  # Impostiamo il nome della label sull'asse delle ascisse

    plt.savefig('../../DataSet/' + DatasetName + '/Output/RandomForest/Confusion Matrix_RandomForest.png')


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
    plt.savefig('../../DataSet/' + DatasetName + '/Output/Random_Forest/Confusion Matrix_Random_Forest.png')


start = time.time()

print('Test Random_Forest: ' + DatasetName)

for test_name in test_labels:
    dir = os.path.join(test_path, test_name)

    # Label di training corrente
    current_label = test_name

    # iterazione delle immagini in ogni sottocartella
    for file in os.listdir(dir):
        file = dir + "/" + os.fsdecode(file)

        # lettura delle immagini e ridimensionamento in una dimensione fissa
        image = cv2.imread(file)

        if image is not None:  # Calcolo delle features per ogni immagine
            image = cv2.resize(image, fixed_size)

            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            fv_histogram = fd_histogram(image)

        # concatenazione delle feature globali in uno Stack
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # Aggiornamento delle liste delle label e dei feature vectors
        labels_test.append(current_label)
        global_features_test.append(global_feature)

    print("[STATUS] Cartella Processata: {}".format(current_label))

# Codifica delle etichette di destinazione con un valore compreso tra 0 e n_classes-1.
targetNames_test = np.unique(labels_test)
le = LabelEncoder()  # Nel nostro caso da 0 a 10, per 11 Periodi Artistici
target_test = le.fit_transform(labels_test)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features_test = scaler.fit_transform(global_features_test)

global_features_test = np.array(rescaled_features_test)
global_labels_test = np.array(target_test)

print("[STATUS] Fine dell'estrazione delle Features del Training Set")

clf = loadModel()
predict(clf)

end = time.time()
print("\nTempo impiegato: circa " + str(
    (round((end - start) / 60))) + " min\nTrain DataSet: " + DatasetName + '\nEsecuzione Terminata')
