import sys
import time
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import os
import sys

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# ******** Dataset_F_Random_20pz ************
number = 20  # Pezzi SLIC--> Dataset_F_Random
DatasetName = 'Dataset_F_Random_' + str(number) + 'pz'

# ******** Dataset_F_Aware_10_cluster ************
# number = 10  # Cluster --> Dataset_F_Aware
# DatasetName = 'Dataset_F_Aware_' + str(number) + '_cluster'

n_Dataset = 1
# ******** Dataset_F_RandomAware ************
DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

if os.path.isdir('../../DataSet/' + DatasetName + '/Output/Classifier'):
    print("Loading..")
else:
    os.mkdir('../../DataSet/' + DatasetName + '/Output/Classifier')

main_directory_dataset = '../../DataSet/' + DatasetName + '/Train&Test/'
batch_size = 32
target_names = ['EGIZIA', 'ETRUSCA', 'GRECA', 'PREISTORICA', 'ROMANA', 'BIZANTINA', 'CUBISMO', 'IMPRESSIONISMO', 'RINASCIMENTO', 'GOTICO', 'SURREALISMO']

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

f = open('../../DataSet/'+DatasetName+ '/Output/Classifier/Log Test_Classifier.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)


def loadModel():
    # carichiamo il modello salvato
    classifier = load_model('../../DataSet/'+DatasetName+ '/Output/Classifier/model_Classifier.h5')
    print("Caricamento modello")

    return classifier

# funzione per disegnare la matrice di confusione
def plot_confusion_matrix(cm, classes, normalize=False, title='Matrice di confusione', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # boh
    plt.title(title)    # Impostiamo il titolo del grafico
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 20, fontsize = 5) # Impostiamo le label sull'asse delle ascisse
    plt.yticks(tick_marks, classes, fontsize = 7) # Impostiamo le label sull'asse delle ordinate

    # Controlliamo se la matrice di confusione Ã¨ normalizzata
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

    plt.savefig('../../DataSet/'+DatasetName+ '/Output/Classifier/Confusion Matrix_Classifier.png')


def predict(test_set):
    # calcoliamo le predizioni
    Y_pred = classifier.predict(test_set)
    y_pred = np.argmax(Y_pred, axis=1)

    # calcoliamo la matrice di confusione
    confusion_Matrix = confusion_matrix(test_set.classes, y_pred)

    # stampiamo i risultati
    print(classification_report(test_set.classes, y_pred, target_names=target_names))
    plot_confusion_matrix(confusion_Matrix, target_names)  # funzione per disegnare la matrice di confusione


start = time.time()

print('Test Model: '+ DatasetName)
# ridimensioniamo otteniamo il dataset di test
test_datagen = ImageDataGenerator(rescale=1. / 255)

# carichiamo il dataset
test_set = test_datagen.flow_from_directory(main_directory_dataset + 'Test_Set', target_size=(256, 256),
                                            batch_size=batch_size,
                                            class_mode='categorical', shuffle=False)

classifier = loadModel()

# valutiamo l'accuratezza del modello
valutazione = classifier.evaluate_generator(test_set, verbose=1)
print("L'accuratezza del modello Ã¨ ", valutazione[1], "Il valore di loss Ã¨ ", valutazione[0])

predict(test_set)

end = time.time()
print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")