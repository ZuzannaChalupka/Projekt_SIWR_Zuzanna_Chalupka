import os
from pathlib import Path
import argparse
import re
import numpy as np
import cv2
import math


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

    #zmienne potrzebne do wzoru:
    name_img_cur = '^' + str(zmianna_pomocnicza)
    current_photo_flag = True
    ilosc = 0
    punkty = []


    nazwa_zdj = None
    liczba_bb_zdj = None



    #lista przechowująca dane dotyczące jednego zdjęcia

    for line in lines:
        if current_photo_flag:
            result = re.match(name_img_cur, line)
            if result:
                current_photo_flag = False
                pp = line[:-1]
                nazwa_zdj = str(data_dir) + '/frames/' + str(pp)
                img = cv2.imread(nazwa_zdj)
                punkty.clear()
                # histogramy.clear()
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
                    cv2.imshow("wycinek", wycinek)
                    cv2.waitKey()
                    print("weszlo")
                    print(ilosc)
                    #probaPROBBBABAAAAA


                    if ilosc == 0:
                        current_photo_flag = True






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    args = parser.parse_args()
    images_dir = Path(args.images_dir)

    read(images_dir)

















