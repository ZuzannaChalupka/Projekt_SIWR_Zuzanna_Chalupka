import os
from pathlib import Path
import argparse
import re
import numpy as np
import cv2
import math
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
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

#źródło: https://www.geeksforgeeks.org/combinations-in-python-without-using-itertools/
#poniższa funkcja jest cała zabrana z punktu Python3 z linku
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
                # print("weszlo")
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

                    # cv2.imshow("wycinek", wycinek)
                    # cv2.waitKey()
                    # print("weszlo")
                    # print(ilosc)
                    """
                    W programie będą użyte takie wartości jak długość, wysokość i przekątna boundingboxa. 
                    W tym celu wykorzystałam taki układ:
                            (x;y)...........((x+w);(y))
                              :                   :
                              :                   :
                          ((x);(y+h)).......((x+w);(y+h))
                    Powyższy prostokąt jest przykładem jak reprezentowane są współrzędne do obliczenia długości Euklidesowej.

                    """
                    #oliczanie szerokości, wysokości i przekątnej bb
                    w_przekatna = math.sqrt(pow((x - x - w), 2) + pow((y - h - y),2))  # odległość euklidesowa, odejmowanie x od x specjalnie, nie zostało zoptymalizowane, żeby było widać jak powstał wzór
                    w_szerokosc = math.sqrt(pow((x - x - w), 2))
                    w_wysokos = math.sqrt(pow((y - h - y), 2))

                    #wpisanie wymiarów do klas
                    przekatna.append(w_przekatna)
                    szerokosc.append(w_szerokosc)
                    wysokosc.append(w_wysokos)

                    #Obliczenie 1/3 wysokości i szerokości
                    w_do_hist_sz = w_szerokosc / 3
                    w_do_hist_wys = w_wysokos / 3

                    #Pomnijeszenie wycinków z każdej strony o 1/3, w celu usięcia tła z histogramó
                    wycinek_do_hit = img[int(y + w_do_hist_wys):int(y + h - w_do_hist_wys), int(x+ w_do_hist_sz):int(x + w - w_do_hist_sz)]
                    # print(w_przekatna)
                    # print(w_szerokosc)
                    # print(w_wysokos)
                    # cv2.imshow("bbbb", wycinek_do_hit)
                    # cv2.waitKey()

                    #histogram do wycinku mniejszego
                    histg = cv2.calcHist([wycinek_do_hit], [0], None, [256], [0, 256])
                    histogramy.append(histg)

                    # plt.plot(histg)
                    # # plt.xlim([0, 256])
                    # plt.show()
                    # cv2.waitKey()



                    if ilosc == 0:
                        current_photo_flag = True
                        # wpisanie parametrów do klasy
                        do_klasy = BBox(punkty, boxy, wysokosc, szerokosc, przekatna, histogramy, nazwa_zdj, liczba_bb_zdj)
                        photos.append(do_klasy)



#Funkcja porównująca histogramy
#na podstawie: https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
def porownaj_his(zdj_1, bb_zdj_1, zdj_2, bb_zdj_2):
    porownaj = cv2.compareHist(photos[zdj_1].histogramy_class[bb_zdj_1], photos[zdj_2].histogramy_class[bb_zdj_2],
                               cv2.HISTCMP_BHATTACHARYYA)
    prawdopo_zgodnosci = \
    cv2.matchTemplate(photos[zdj_1].histogramy_class[bb_zdj_1], photos[zdj_2].histogramy_class[bb_zdj_2],
                      cv2.TM_CCOEFF_NORMED)[0][0]
    wynik_his = 1 - prawdopo_zgodnosci
    wynik_his_10 = (porownaj / 3) + wynik_his  # wykorzystanie 33% z porówania his, ponieważ jest mniej dokładne od metody "wzornikowej"

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



#Funkcja wypisująca -1 dla pierwszego zdjęcia
def prawdopodobienstwo_dla_1_bb():
    linia_pierwsza = None
    # Wypisanie -1 dla liczby bb na 1 zdj
    for lic_bb in range(photos[0].liczba_BB):
        if not linia_pierwsza:
            linia_pierwsza = '-1'
        else:
            linia_pierwsza = '-1'
    print(linia_pierwsza)

#Graf cały powstał na podstawie dokumentacji do biblioteki: https://pgmpy.org/models/factorgraph.html
def prawdopodienstwo():
    wynik = []
    flaga = False


    #pętla do przejścia przez wszytkie zdjęcia, oprócz 1 wraz z ich nr id zdjecia
    for x, bb in enumerate(photos[1:]):

        Graf = FactorGraph()

        for box in range(bb.liczba_BB):
            nazwa_zdj = bb.nazwa + '_' + str(box)
            Graf.add_node(nazwa_zdj) # dodanie nowego "węzła" oraz aktualizacja
            #pętla po bb z poprzedniego zdjęcia
            for bb_minus_1 in range(photos[x].liczba_BB):
                #użycie funkcji do porównania histogramói i wymiarów
                prawdop_po_his = porownaj_his(x, bb_minus_1, x+1, box)
                p_h, p_w, p_p = porow_wymiary(x, bb_minus_1, x+1, box)
                #dodanie współczynników z większa wagą dla wysokości i długości
                suma_prawd = (prawdop_po_his + 2 *p_h + p_p + 2 *p_w) / 6
                wynik.append(suma_prawd)

            x1 = DiscreteFactor([nazwa_zdj], [len(wynik)+1], [[0.5]+wynik])
            Graf.add_factors(x1)
            Graf.add_node(x1)
            Graf.add_edge(nazwa_zdj, x1)
            wynik.clear()
            # print("weszpoweszlo")
            if bb.liczba_BB >1:
                flaga = True
                # print("wesz")

        if flaga:
            # print("weszlo")
            zmienna = []
            y1 = np.ones((photos[x].liczba_BB+1, photos[x].liczba_BB+1 ))
            y2 = np.eye(photos[x].liczba_BB+1)
            # print("jestem tu")
            w = y1 - y2
            w[0][0] += 1

            # print("print1", bb.liczba_BB)
            # print(range(bb.liczba_BB))
            for j in range(bb.liczba_BB):
                # print("tutututututu")
                # print("print2", bb.liczba_BB)
                nazwa_zdj = bb.nazwa + '_' + str(j)
                zmienna.append(nazwa_zdj)
            zmienna_1 = [x for x in preparation(zmienna, 2)]
            # print("cccccccc", zmienna_1)
            for j in range(len(zmienna_1)):

                x2 = DiscreteFactor([zmienna_1[j][0], zmienna_1[j][1]], [photos[x].liczba_BB + 1, photos[x].liczba_BB + 1], w)
                # print("weszło2")
                # print("print", x2)
                cv2.waitKey()
                Graf.add_factors(x2)
                Graf.add_node(x2)
                Graf.add_edges_from([(zmienna_1[j][0], x2), (zmienna_1[j][1], x2)])
            flaga = False
        # print(Graf)

        result = None
        belief_propagation= BeliefPropagation(Graf)
        belief_propagation.calibrate()

        map = belief_propagation.map_query(Graf.get_variable_nodes(), show_progress=False)
        # print("map", map)
        for i in range(bb.liczba_BB):  # zapisanie przypisanych bb do zmiennej, aby potem wyświetlić je w jednej linii
            nazwa_zdj = bb.nazwa + '_' + str(i)
            Liczby_z_map = map[nazwa_zdj] - 1
            # print(Liczby_z_map)
            if not result:
                # print(Liczby_z_map)
                result = str(Liczby_z_map)
            else:
                result = result + ' ' + str(Liczby_z_map)
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)

    read(images_dir)
    prawdopodobienstwo_dla_1_bb()
    prawdopodienstwo()

















