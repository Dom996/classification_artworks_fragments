import time
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from matplotlib import pyplot as plt
import sys

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

n_Dataset = 1
# ******** Dataset_F_RandomAware ************
DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

# ******** Dataset_F_Random_20pz ************
number = 20  # Pezzi SLIC--> Dataset_F_Random
DatasetName = 'Dataset_F_Random_' + str(number) + 'pz'

# ******** Dataset_F_Aware_10_cluster ************
# number = 10 #Cluster --> Dataset_F_Aware
# DatasetName = 'Dataset_F_Aware_' + str(number) + '_cluster'

if os.path.isdir('../../DataSet/' + DatasetName + '/Output/Classifier'):
    print("Loading..")
else:
    os.mkdir('../../DataSet/' + DatasetName + '/Output/Classifier')

main_directory_dataset = '../../DataSet/' + DatasetName + '/Train&Test/'
batch_size = 16


class ConsoleOutput(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


f = open('../../DataSet/' + DatasetName + '/Output/Classifier/Log Train_Classifier.txt', "w")
original = sys.stdout
sys.stdout = ConsoleOutput(sys.stdout, f)


def create_model():
    # Inizializza il classificatore
    classifier = Sequential()

    # Aggiungiamo uno strato convoluzionale alla CNN
    # nb_filters = 64 sono il numero di feature maps che vogliamo creare
    # nb_rows e nb_columns = (3,3) sono la dimensione della riga e della colonna del filtro
    # input_shape il formato previsto della matrice in input
    # activation = 'relu' rappresenta la funzione di attivazione
    classifier.add(Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))

    # Aggiungiamo uno strato di Max Pooling
    classifier.add(MaxPooling2D(pool_size=(3, 3)))

    # Aggiungiamo uno strato convoluzionale alla CNN
    classifier.add(Conv2D(32, (3, 3), activation='relu'))

    # Aggiungiamo uno strato di Max Pooling
    classifier.add(MaxPooling2D(pool_size=(3, 3)))

    # Aggiungiamo uno strato Flatten
    classifier.add(Flatten())

    # Aggiungiamo un Hidden Layer
    # Con units = 256 rappresentiamo i dati in output dell'Hidden Layer
    classifier.add(Dense(units=256, activation='relu'))

    # Ultimo strato del nostro modello
    # units = 11 rappresentiamo i dati in output dell'Output Layer (11 Periodi Storici)
    classifier.add(Dense(units=11, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    # loss = 'categorical_crossentropy' rappresenta la funzione di loss utilizzata dal modello
    # metrics=['accuracy'] rappresenta la metrica di score utilizzata dal modello
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.summary()

    return classifier


def plot_history(classifier):
    loss_list = [s for s in classifier.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in classifier.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in classifier.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in classifier.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    epochs = range(1, len(classifier.history[loss_list[0]]) + 1)

    figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ************ Training Loss & Training Accuracy ************
    # Training Loss
    for l in loss_list:
        ax1.plot(epochs, classifier.history[l], 'b',
                 label='Training Loss (' + str(str(format(classifier.history[l][-1], '.5f')) + ')'))
    # Training Accuracy
    for l in acc_list:
        ax1.plot(epochs, classifier.history[l], 'r',
                 label='Training Accuracy (' + str(format(classifier.history[l][-1], '.5f')) + ')')
    ax1.set_title('Training Loss & Training Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    # ************ Validation Accuracy & Validation Loss ************
    # Validation Accuracy
    for l in val_acc_list:
        ax2.plot(epochs, classifier.history[l], 'b',
                 label='Validation Accuracy (' + str(format(classifier.history[l][-1], '.5f')) + ')')
    # Validation Loss
    for l in val_loss_list:
        ax2.plot(epochs, classifier.history[l], 'g',
                 label='Validation Loss (' + str(str(format(classifier.history[l][-1], '.5f')) + ')'))

    ax2.set_title('Validation Accuracy & Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    plt.savefig('../../DataSet/' + DatasetName + '/Output/Classifier/Accuracy VS Loss_Classifier.png')


def classifier(classifier):
    model_Classifier = classifier.fit(training_set, epochs=20, validation_data=test_set, shuffle=True)

    plot_history(model_Classifier)  # Creiamo i grafici del training
    classifier.save(
        '../../DataSet/' + DatasetName + '/Output/Classifier/model_Classifier.h5')  # salviamo il modello dopo aver effettuato il training
    print("Modello salvato")


start = time.time()
print('Train Model: ' + DatasetName)

# Normalizzazione Dataset
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Carichiamo il Training Set
training_set = train_datagen.flow_from_directory(main_directory_dataset + 'Training_Set', target_size=(256, 256),
                                                 batch_size=batch_size, class_mode='categorical')

# Carichiamo il Test Set
test_set = test_datagen.flow_from_directory(main_directory_dataset + 'Test_Set', target_size=(256, 256),
                                            batch_size=batch_size, class_mode='categorical')

model = create_model()  # Creiamo il modello
classifier(model)  # Alleniamo e salviamo il modello

end = time.time()
print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")
