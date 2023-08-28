import os
import time
from os import listdir
from PIL import Image as PImage
from PIL import ImageOps
from skimage.segmentation import slic
from skimage.util import img_as_float
import numpy as np
import cv2
from matplotlib import cm
import sys

number_fragments = 10  # Numero approssimativo dei frammenti da dare in output

# ******** Dataset_F_RandomAware ************
n_Dataset = 1
DatasetName = 'Dataset_F_RandomAware_' + str(n_Dataset)

# ******** Dataset_F_Random ************
# DatasetName = 'Dataset_F_Random_' + str(number_fragments) + 'pz'

# path = 'Train'
path = "Test"

PATH_main_dataset = "../Main Dataset/" + path  # directory dove è contenuta il dataset originale
PATH_dataset_F_Random = '../DataSet/' + DatasetName  # direcotry dove sono contenuti i frammenti randomici

PATH = [PATH_dataset_F_Random, PATH_dataset_F_Random + '/Train&Test',
        PATH_dataset_F_Random + '/Output',
        PATH_dataset_F_Random + '/Train&Test/Train',
        PATH_dataset_F_Random + '/Train&Test/Test']
for pathDir in PATH:
    if os.path.isdir(pathDir):
        print("Loading..")
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


f = open(PATH_dataset_F_Random + '/Output/Log Frammentazione Dataset_F_Random_' + str(number_fragments) + 'pz.txt', "w")
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)


def load_art_folder():
    artistic_period = []  # lista che conterrà i periodi artistici
    # il seguente for ci permette di ottenere le directory di ogni periodo artistico
    for filepath in os.listdir(PATH_main_dataset):
        # ogni directory dei periodi artistici viene aggiunta alla lista artistic_period
        artistic_period.append(filepath)
    return artistic_period  # la funzione restituisce una lista delle directory dei periodi artistici


# compactness: Questo parametro dipende fortemente dal contrasto dell'immagine e dalle forme degli oggetti nell'immagine.
# Valori più alti danno luogo a segmenti più squadrati/cubici di forma sempre più regolare.

# sigma: larghezza del kernel di smoothing gaussiano per la pre-elaborazione per ogni dimensione dell'immagine.
def segment_image(image):
    segment = slic(img_as_float(image), number_fragments, sigma=10)
    return segment  # la funzione restituisce i segmenti delle immagini


def extract_fragment(segments, original_image, folder, image_name):
    image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    count = 0
    for (i, segVal) in enumerate(np.unique(segments)):
        # costruiamo una maschera per il segmento selezionato
        mask = np.zeros(image.shape[:2], dtype="uint8")
        # coloriamo di bianco la parte della maschera relativa al segmento selezionato
        mask[segments == segVal] = 255

        # troviamo i contorni della parte bianca della maschera e salviamo queste dimensioni nella variabile "cnts"
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # calcoliamo il bounding box del segmento ed estraiamo la regione d'interesse
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            ROI = image[y:y + h, x:x + w]
            ROI_mask = mask[y:y + h, x:x + w]
            break

        # Convertiamo l'immagine in immagine PIL
        imageCv2_to_convert = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
        pil_image = PImage.fromarray(imageCv2_to_convert)

        # convertiamo la maschera in immagine PIL
        maschera = PImage.fromarray(np.uint8(cm.gist_earth(ROI_mask) * 255))

        # convertiamo la maschera in scala di grigi
        mask2 = maschera.convert('L')

        # ritagliamo la regione d'interesse dall'immagine originale
        final_image = ImageOps.fit(pil_image, mask2.size)
        # modifichiamo il canale alpha dell'immagine ottenuta in precedenza
        final_image.putalpha(mask2)
        count = count + 1

        # salviamo l'immagine con file PNG
        im1 = image_name.split('.')[0]  # es. "image_name = 1.jpg" im1 = 1
        nomeDef = folder + "_R" + im1
        final_image.save(PATH_dataset_F_Random + '/Train&Test/' + path + '/' + folder + '/' + nomeDef + '.' + str(count) + '.png')


start = time.time()

print('Frammentazione Randomica DataSet: ' + DatasetName + '\n')

artistic_period = load_art_folder()
i = 1

for folder in artistic_period:  # Per ogni periodo storico
    start_folder = time.time()

    if os.path.isdir(PATH_dataset_F_Random + '/Train&Test/' + path + '/' + folder):
        print("Loading..")
    else:
        os.mkdir(PATH_dataset_F_Random + '/Train&Test/' + path + '/' + folder)

    print(folder + ": Inizio Frammentazione")
    imagesList = listdir(PATH_main_dataset + '/' + folder + "/")  # Crea una lista di immagini

    for image in imagesList:  # Per ogni immagine nella lista
        img = PImage.open(PATH_main_dataset + '/' + folder + "/" + image)
        segments = segment_image(img)  # Segmenta l'immagine
        extract_fragment(segments, img, folder, str(image))  # Estrae Frammenti Randomici
    end_folder = time.time()
    print("Tempo impiegato: circa " + str((round((
                                                         end_folder - start_folder) / 60))) + " min\n" + folder + ": Frammentazione Completata\nCartelle Frammentate: " + str(
        i) + "/11\n")
    i += 1

end = time.time()
print("Tempo impiegato: circa " + str((round((end - start) / 60))) + " min\nEsecuzione Terminata")
