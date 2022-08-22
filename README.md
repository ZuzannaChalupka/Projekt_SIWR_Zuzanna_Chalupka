# Projekt_SIWR_Zuzanna_Chalupka
Projekt polega na stworzeniu systemu śledzącego pieszych na podstawie stworzonego 
Projekt zakłada stworzenie systemu śledzącego przechodniów z wykorzystaniem probabilistycznych modeli graficznych. System przeznaczony jest do określania lokalizacji przechodniów w kolejnych kadrach kamery poprzez przypisywanie poszczególnym osobom Bounding Boxów.

Wczytane z pliku bounding boxy są nakładane na zdjęcie a następnie są wycinane i wpisywane do klasy, w której przechowujemy dane potrzebne do stworzenia modelu. 
Do klasy również wpisywane są takie parametry jak: wymiary bb (szerokość, wysokość i przekątna) oraz histogramy pomniejszonych. <br />

W programie zostało użyte wiele komentarzy, które tłumacza poszczególne zmienne i funkcje oraz linki do wykorzystanych fragmentów kodu.  

# Graf 
![graf](https://user-images.githubusercontent.com/50628242/185988465-2a556f92-ac76-4122-bbc0-023b6e6c3b95.png)

Graf wykorzystuje: 
* porówanie histogramów pomniejszonych o bounding boxów, w celu unikania wyrywania tła, 
* porówanie wielkości bounding boxów oraz ich przekątnych. 

Zmienne te składają się na podobieństwa bounding boxów z obecnego zdjęcia do tych z poprzedniego. Im większe podobieństwo tym większe prawdopodobieństwo na to, że osoba z poprzedniego zdjęcia pasuje do osoby z obecnego. 

# Uruchomienie programu 
W celu poprawnego uruchomienia programu należy w terminalu wpisać <br />
$ python .\main.py path/to/photos





