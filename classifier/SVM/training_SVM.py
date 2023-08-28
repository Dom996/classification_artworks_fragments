import sys
import sys
import time

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
import pickle as pk

# n_Dataset = 1
# ******** Dataset_F_RandomAware ************
# DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

# ******** Dataset_F_Random_20pz ************
# number = 20 #Pezzi SLIC--> Dataset_F_Random
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


f = open('../../DataSet/' + DatasetName + '/Output/SVM/Log SVM_Training.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

main_directory_dataset = '../../DataSet/' + DatasetName + '/Train&Test/'
train_path = main_directory_dataset + 'Training_Set'

# bins for histograms
bins = 8


# features description -1:  Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor -2: Haralick Texture
def fd_haralick(image):
    # conver the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ccompute the haralick texture fetature ve tor
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic


# feature-description -3: Color Histogram
def fd_histogram(image, mask=None):
    # conver the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # COPUTE THE COLOR HISTPGRAM
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histog....
    return hist.flatten()


start = time.time()
print('Train SVM: ' + DatasetName)

# get the training data labels
train_labels = os.listdir(train_path)

# sort  training labels and test labels
train_labels.sort()

# empty list to hold feature vectors and labels
global_features = []
labels = []
# make a fix file size
fixed_size = tuple((700, 700))
# lop over the training data sub folder
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

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
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

global_features = np.array(rescaled_features)
global_labels = np.array(target)

print("[STATUS] end of features extraction of training set")

# create the model - Random Forests
# clf = RandomForestClassifier(n_estimators=400)
clf = svm.LinearSVC()

# fit the training data to the model
clf.fit(global_features, global_labels)
with open('../../DataSet/' + DatasetName + '/Output/SVM/model_SVM.h5', 'wb') as f:
    pk.dump(clf, f)
print('Salvataggio Modello')

end = time.time()
print("\nTempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")
