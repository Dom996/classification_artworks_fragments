import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D,Flatten,Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import sys
import os
import time

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# ******** Dataset_F_Random_20pz ************
number = 20 #Pezzi SLIC--> Dataset_F_Random
DatasetName = 'Dataset_F_Random_' + str(number) + 'pz'

# ******** Dataset_F_Aware_10_cluster ************
#number = 10 #Cluster --> Dataset_F_Aware
#DatasetName = 'Dataset_F_Aware_' + str(number) + '_cluster'

n_Dataset = 1
# ******** Dataset_F_RandomAware ************
DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

if os.path.isdir('../../DataSet/'+DatasetName+'/Output/AlexNet'):
    print ("Loading..")
else:
    os.mkdir('../../DataSet/'+DatasetName+'/Output/AlexNet')

main_directory_dataset = '../../DataSet/'+DatasetName+'/Train&Test/'
batch_size = 32

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

f = open('../../DataSet/'+DatasetName+ '/Output/AlexNet/Log Train_AlexNet.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)


def alexnet():
    # Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(256, 256, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(256*256*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(11)) #11 CLASSI
    model.add(Activation('softmax'))

    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])

    return model

def plot_history(classifier):
    loss_list = [s for s in classifier.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in classifier.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in classifier.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in classifier.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
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

    plt.savefig('../../DataSet/'+ DatasetName + '/Output/AlexNet/Accuracy VS Loss_AlexNet.png')


def classifier(classifier):
    model_Classifier = classifier.fit(training_set, epochs=2, validation_data=test_set, shuffle=True)

    plot_history(model_Classifier)

    classifier.save('../../DataSet/'+DatasetName+ '/Output/AlexNet/model_AlexNet.h5')
    print("Modello salvato")

start = time.time()

print('Train AlexNet: '+ DatasetName)
#Normalizzazione Dataset
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(main_directory_dataset+'Training_Set', target_size=(256, 256), batch_size=batch_size, class_mode='categorical')

test_set = test_datagen.flow_from_directory(main_directory_dataset+'Test_Set', target_size=(256, 256), batch_size=batch_size, class_mode='categorical')

model = alexnet()
classifier(model)

end = time.time()
print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")