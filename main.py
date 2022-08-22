import os
from pathlib import Path
import argparse
import re
import numpy as np
import cv2
import math
from pgmpy.models import FactorGraph
from matplotlib import pyplot as plt

#Pamietaj: dodaj linijki w ktorych to jest
photos = []

class BBox:
    def __init__(self, punkt, boxy_class, wysokosc_class, szerokosc_class, przekatna_class, histogramy_class, nazwa, liczba_BB ):
        self.punkt = list(punkt)                        #lista z punktami do bounding box
        self.boxy_class = list(boxy_class)              #lista z bounding boxami wycietymi
        self.wysokosc_class = list(wysokosc_class)      #lista z wysokosciami bb
        self.szerokosc_class = list(szerokosc_class)    #lista z szerokosciami bb
        self.przekatna_class = list(przekatna_class)    #lista z przekatnymi bb
        self.histogramy_class = list(histogramy_class)  #lista z histogramami
        self.nazwa = nazwa                              #nazwa zdjecia
        self.liczba_BB = liczba_BB                      #liczba bb na jednym zdjęciu

def preparation(search, r):
    name = search
    length = len(name)
    if r> length:
        return
    id = np.arange(r)
    yield tuple(name[i] for i in id)
    while True:
        for i in reversed(range(r)):
            if id[i] != i + length - r:
                break
            else:
                return

            id[i] +=1
            for ii in range(i+1, r):
                id[ii] = id[ii-1]+1
            yield tuple(name[i] for i in id)

def read(data_dir):
    plik = str(data_dir) + '/bboxes.txt'
    with open(plik) as f:
        lines = f.readlines()
    zmianna_pomocnicza = None

    for line in lines:
        zmianna_pomocnicza = line[:1]
        break

    #zmienne potrzebne do wzoru, opisane w większości przy definiowaniu klasy:
    name_img_cur = '^' + str(zmianna_pomocnicza)
    current_photo_flag = True
    ilosc = 0
    punkty = []
    boxy = []
    wysokosc = []
    szerokosc = []
    przekatna = []
    histogramy = []

    nazwa_zdj = None
    liczba_bb_zdj = None



    for line in lines:
        if current_photo_flag:
            result = re.match(name_img_cur, line)
            if result:
                #czyszczenie zmiennych przed wpisaniem do nowego zdjęcia
                current_photo_flag = False
                pp = line[:-1]
                nazwa_zdj = str(data_dir) + '/frames/' + str(pp)
                img = cv2.imread(nazwa_zdj)
                punkty.clear()
                boxy.clear()
                wysokosc.clear()
                szerokosc.clear()
                przekatna.clear()
                histogramy.clear()
                # odleglosci.clear()
                print("weszlo")
        else:
            if len(line)<3 and line != '\n':
                ilosc = int(line) #ile mamy wczytac kolejnych lini
                liczba_bb_zdj = ilosc
                # print("weszlo2")
            else:
                if ilosc !=0:
                    dane = line.split()
                    punkty.append(float(dane[0]))
                    punkty.append(float(dane[1]))
                    punkty.append(float(dane[2]))
                    punkty.append(float(dane[3]))
                    x = float(dane[0])  #x1
                    y = float(dane[1])  #y1
                    w = float(dane[2])  #x2 = (x+w)
                    h = float(dane[3])  #y2 = (y+h)

                    ilosc-=1

                    #wycinanie boundingboxa z osobą
                    wycinek = img[int(y):int(y + h), int(x):int(x + w)]

                    #wpisanie wyciętego BB do klasy
                    boxy.append(wycinek)

                    cv2.imshow("wycinek", wycinek)
                    cv2.waitKey()
                    print("weszlo")
                    print(ilosc)
                    """
                    W programie będą użyte takie wartości jak długość, wysokość i przekątna boundingboxa. 
                    W tym celu wykorzystałam taki układ:
                            (x;y)...........((x+w);(y))
                              :                   :
                              :                   :
                          ((x);(y+h)).......((x+w);(y+h))
                    Powyższy prostokąt jest przykładem jak reprezentowane są współrzędne do obliczenia długości Euklidesowej.

                    """
                    w_przekatna = math.sqrt(pow((x - x - w), 2) + pow((y - h - y),2))  # odległość euklidesowa, odejmowanie x od x specjalnie, nie zostało zoptymalizowane, żeby było jak powstał wzór
                    w_szerokosc = math.sqrt(pow((x - x - w), 2))
                    w_wysokos = math.sqrt(pow((y - h - y), 2))

                    #wpisanie wymiarów do klas
                    przekatna.append(w_przekatna)
                    szerokosc.append(w_szerokosc)
                    wysokosc.append(w_wysokos)

                    w_do_hist_sz = w_szerokosc / 3
                    w_do_hist_wys = w_wysokos / 3

                    wycinek_do_hit = img[int(y + w_do_hist_wys):int(y + h - w_do_hist_wys),
                                     int(x + w_do_hist_sz):int(x + w - w_do_hist_sz)]
                    # print(w_przekatna)
                    # print(w_szerokosc)
                    # print(w_wysokos)
                    cv2.imshow("bbbb", wycinek_do_hit)
                    cv2.waitKey()

                    # #histogram do wycinku
                    histg = cv2.calcHist([wycinek], [0], None, [256], [0, 256])
                    histogramy.append(histg)

                    # plt.plot(histg)
                    # plt.xlim([0, 256])
                    # plt.show()
                    # cv2.waitKey()



                    if ilosc == 0:
                        current_photo_flag = True
                        do_klasy = BBox(punkty, boxy, wysokosc, szerokosc, przekatna, histogramy, nazwa_zdj, liczba_bb_zdj)
                        photos.append(do_klasy)


#Funkcja porównująca histogramy
#na podstawie: https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
#
def porow_histogram(zdj_1, bb_zdj_1, zdj_2, bb_zdj_2):

    porownaj = cv2.compareHist(photos[zdj_1].histogramy_class[bb_zdj_1], photos[zdj_2].histogramy_class[bb_zdj_2],
                                cv2.HISTCMP_BHATTACHARYYA)
    prawdopo_zgodnosci = cv2.matchTemplate(photos[zdj_1].histogramy_class[bb_zdj_1], photos[zdj_2].histogramy_class[bb_zdj_2],
                                               cv2.TM_CCOEFF_NORMED)[0][0]
    wynik_his = 1 - prawdopo_zgodnosci
    wynik_his_10 = (porownaj / 10) + wynik_his  # wykorzystanie 10% z porówania his, ponieważ jest mniej dokładne od metody "wzornikowej"

    # print("to czego szuka" )
    # print(1-wynik_his_10)
    return 1 - wynik_his_10


def porow_wymiary(zdj_1, bb_zdj_1, zdj_2, bb_zdj_2):
    #Wysokość 1 i 2 bb
    H_bb_1 = photos[zdj_1].wysokosc_class[bb_zdj_1]
    H_bb_2 = photos[zdj_2].wysokosc_class[bb_zdj_2]

    #Szerokość 1 i 2 bb
    W_bb_1 = photos[zdj_1].szerokosc_class[bb_zdj_1]
    W_bb_2 = photos[zdj_2].szerokosc_class[bb_zdj_2]

    #Przekątna 1 i 2 bb
    D_bb_1 = photos[zdj_1].przekatna_class[bb_zdj_1]
    D_bb_2 = photos[zdj_2].przekatna_class[bb_zdj_2]

    # stosunki wysokość, przekątnych i szerokości, w celu uzyskania liczb mniejszych niż 1 dodany jest warunek
    if H_bb_1 > H_bb_2:
        stosunek_H = H_bb_2/H_bb_1
    else:
        stosunek_H = H_bb_1 / H_bb_2

    if W_bb_1 > W_bb_2:
        stosunek_W = W_bb_2/W_bb_1
    else:
        stosunek_W = W_bb_1/W_bb_2

    if D_bb_1 > D_bb_2:
        stosunek_D = D_bb_2/D_bb_1
    else:
        stosunek_D = D_bb_1/D_bb_2

    return stosunek_H, stosunek_W, stosunek_D


def prawdopodienstwo():
    flaga = False
    linia_pierwsza = None
    
    # Wypisanie -1 dla liczby bb na 1 zdj
    for lic_bb in range(photos[0].liczba_BB):
        if not linia_pierwsza:
            linia_pierwsza = '-1'
        else:
            linia_pierwsza = '-1'
    print(linia_pierwsza)
#
#     #pętla do przejścia przez wszytkie zdjęcia wraz z ich nr id zdjecia
#     for bb in enumerate(photos[1:]):
#         Graf = FactorGraph()
#         for box in range(bb.liczba_BB):
#             nazwa_zdj = bb.nazwa + '_' + str(box)
#
#             #pętla po bb z poprzedniego zjęcia
#             for bb_minus_1 in range(photos[bb].liczba_BB):
#                 prawdop_po_his = porow_histogram(bb_minus_1, box)




















if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)

    read(images_dir)

















