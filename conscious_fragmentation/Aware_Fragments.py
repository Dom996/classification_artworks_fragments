import sys
import time
import cv2
from PIL import Image, ImageDraw, ImageOps
import pandas as pd
from scipy.spatial.qhull import ConvexHull
from sklearn.cluster import KMeans
import numpy as np
import os
from os import listdir

n_cluster = 40  # numero dei cluster desiderati

# ******** Dataset_F_RandomAware************
# n_Dataset = 4
# DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

# ******** Dataset_F_Aware_10 cluster************
# n_features = [800, 800, 500, 400, 500, 500, 900, 600, 700, 700, 800]  # vettore con in numero di features da cercare
# per ogni periodo storico

# ******** Dataset_F_Aware_20 cluster************
# n_features = [1100, 900, 1000, 600, 1500, 700, 1500, 1500, 2500,
#              1200, 2300]  # vettore con in numero di features da cercare per ogni periodo storico

# ******** Dataset_F_Aware_40 cluster************ featurex2
n_features = [3200, 1800, 3400, 1200, 1600, 1400, 3000, 4500, 4300, 2800,
              4600]  # vettore con in numero di features da cercare per ogni periodo storico

# ******** Dataset_F_Aware_80 cluster************ featurex4
# n_features = [8800, 7200, 9600, 4800, 6400, 4200, 12000, 12000,
#              17200, 11200, 18400]  # vettore con in numero di features da cercare per ogni periodo storico

# ******** Dataset_F_Aware************
DatasetName = 'Dataset_F_Aware_' + str(n_cluster) + '_cluster'

path = "Test"
# path = "Train"

PATH_main_dataset = "../Main Dataset/" + path  # directory dove è contenuta il DataSet originale
PATH_dataset_F_Aware = '../DataSet/' + DatasetName  # direcotry dove sono contenuti i frammenti randomici

PATH = [PATH_dataset_F_Aware, PATH_dataset_F_Aware + '/Train&Test', PATH_dataset_F_Aware + '/Train&Test/Train',
        PATH_dataset_F_Aware + '/Train&Test/Test', PATH_dataset_F_Aware + '/Output']

for pathDir in PATH:
    if os.path.isdir(pathDir):
        print('Loading...')
    else:
        os.mkdir(pathDir)


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


f = open(PATH_dataset_F_Aware + '/Output/Log Frammentazione Dataset_F_Aware_' + str(n_cluster) + '_cluster.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)


def load_art_folder():
    artistic_period = []  # lista che conterrà i periodi artistici
    # il seguente for ci permette di ottenere le directory di ogni periodo artistico
    for filepath in os.listdir(PATH_main_dataset):
        # ogni directory dei periodi artistici viene aggiunta alla lista artistic_period
        artistic_period.append(filepath)
    return artistic_period  # la funzione restituisce una lista delle directory dei periodi artistici


def features_extraction(original_img, n_features):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)  # convertiamo l'immagine in scala di grigi
    gray = np.float32(gray)
    # estraiamo le features tramite l'utilizzo dell'algoritmo Harrison Corner
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=n_features, qualityLevel=0.001, minDistance=1)
    x1 = []  # vettore delle coordinate x delle features
    y1 = []  # vettore delle coordinate y delle features
    for item in corners:  # inseriamo i valori delle coordinate nei vettori
        x, y = item[0]
        x1.append(x)
        y1.append(y)

    d = {'x': x1, 'y': y1}
    df = pd.DataFrame(data=d)  # dataframe delle features
    return df


def k_means_clustering(df, n_cluster):
    kmeans = KMeans(n_clusters=n_cluster)  # definiamo i cluster che l'agoritmo K-means dovrà calcolare
    kmeans.fit(df)  # K-means calcola i cluster

    labels = kmeans.predict(df)  # calcoliamo le etichette dei cluster
    df['cluster'] = labels  # associamo le etichette ad ogni feature del dataframe
    return df


def extract_fragments(df, img, image, folder):
    count = 0
    open_cv_image = np.array(img)  # convertiamo l'immagine originale in formato numPy Array
    height, width = open_cv_image.shape[:2]

    for num in range(0, n_cluster):
        # prendiamo le feature appartenenti ad un cluster
        x_cluster = np.array(df["x"][df.cluster == num])
        y_cluster = np.array(df["y"][df.cluster == num])
        features_coordinates = list(zip(x_cluster, y_cluster))

        hull = ConvexHull(np.asarray(features_coordinates))  # calcoliamo il ConvexHull del cluster
        vertices = list()

        for z in hull.vertices:  # prendiamo le coordinate delle features che rappresentano i vertici del ConvexHull
            x_coord = x_cluster[z]
            y_coord = y_cluster[z]
            tupla = (x_coord, y_coord)
            vertices.append(tupla)

        empty_img = Image.new('L', (width, height), 0)  # costruiamo la maschera nera
        ImageDraw.Draw(empty_img).polygon(vertices, outline=1,
                                          fill=255)  # disegnamo il poligono utilizzando i vertici precedenti
        mask = np.array(empty_img)  # convertiamo l'immagine della maschera in formato numPy Array

        # troviamo i contorni della parte bianca della maschera e salviamo queste dimensioni nella variabile "contours"
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # calcoliamo il bounding box del segmento ed estraiamo la regione d'interesse
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ROI_mask = mask[y:y + h, x:x + w]
            ROI = open_cv_image[y:y + h, x:x + w]
            break

        ROI_pil = Image.fromarray(ROI)  # convertiamo immagine ritagliata in formato PIL
        ROI_mask_pil = Image.fromarray(ROI_mask)  # convertiamo la maschera ritagliata in formato PIL
        final_img = ImageOps.fit(ROI_pil,
                                 ROI_mask_pil.size)  # ritagliamo la regione d'interesse dall'immagine originale
        final_img.putalpha(ROI_mask_pil)  # modifichiamo il canale alpha dell'immagine ottenuta in precedenza
        count = count + 1

        # salviamo l'immagine con file PNG
        im1 = image.split('.')[0]  # es. "image_name = 1.jpg" im1 = 1
        # nomeDef = folder + "_" + im1
        nomeDef = folder + "_A" + im1
        final_img.save(
            PATH_dataset_F_Aware + '/Train&Test/' + path + '/' + folder + '/' + nomeDef + '.' + str(count) + '.png')


start = time.time()

print('Frammentazione Consapevole,' + path + ' del DataSet: ' + DatasetName + '\n')

artistic_period = load_art_folder()
j = 0  # indice vettore del numero delle features

dataframe = pd.DataFrame()
for folder in artistic_period:  # Per ogni periodo storico
    start_folder = time.time()

    if os.path.isdir(PATH_dataset_F_Aware + '/Train&Test/' + path + '/' + folder):
        print()
    else:
        os.mkdir(
            PATH_dataset_F_Aware + '/Train&Test/' + path + '/' + folder)  # Crea nuova directory per il peridodo storico di riferimento

    print(folder + ": Inizio Frammentazione\nFeatures: " + str(n_features[j]))
    imagesList = listdir(PATH_main_dataset + '/' + folder + "/")  # Crea una lista di immagini

    for image in imagesList:  # Per ogni immagine nella lista
        img = Image.open(PATH_main_dataset + '/' + folder + "/" + image)
        img_cv2 = np.array(img)
        df_temp = features_extraction(img_cv2, n_features[j])  # estrae features dall'immagine
        dataframe = k_means_clustering(df_temp, n_cluster)  # clustering delle features calcolate in precedenza
        extract_fragments(dataframe, img, str(image), folder)  # frammentazione e salvataggio frammenti
    end_folder = time.time()
    print("Tempo impiegato: circa " + str((round((
                                                         end_folder - start_folder) / 60))) + " min\n" + folder + ": Frammentazione Completata\nCartelle Frammentate: " + str(
        j + 1) + "/11\n")
    j += 1

end = time.time()
print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")
