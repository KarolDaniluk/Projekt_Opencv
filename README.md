# Projekt OpenCV
## Unattended Object Detection with CUDA
Projekt realizuje detekcję pozostawionego bagażu w miejscach publicznych.

## Przykładowe nagrania
Przykładowe pliki znajdują się katalogu [PETS_2006/mini](https://github.com/KarolDaniluk/Projekt_Opencv/tree/master/PETS_2006/mini).
W katalogu znajdują się dwa warianty sytuacji: A i B, nagrane za pomocą dwóch kamer z różnych ujęć.
W celu przyspieszenia obliczeń zostały one pomniejszone z rozmiaru 720x576 do rozdzielczości 400 x 320.

## CUDA
W celu przyspieszenia obliczeń projekt wykorzystuje możliwości kart graficznych Nvidia zgodnych z [technologią CUDA](https://developer.nvidia.com/cuda-zone).
Do tego celu została skompilowana specjalna wersja OpenCV łącznie ze wszystkimi [dodatkowymi modułami](https://github.com/opencv/opencv_contrib).
Gotową paczkę można pobrać z dysku Google: [opencv_cuda_allcontrib_noworld](https://drive.google.com/open?id=1GNEdk11w0Kd6UuqYJDIbUi4cBB-Aoff3)
Żeby z niej skorzystać należy wypakować archiwum (domyślnie katalog C:\\) i dodać znajdujące się w nim foldery **cuda_dll** oraz **\\x64\\vc15\\bin** do zmiennej systemowej PATH.

## Działanie
Detekcja obiektów odbywa się w kilku etapach.

### Pobranie tła
Za pomocą stworzonego obiektu _backgroundSubstractor_ i metody MOG2 z filmu pobierane jest tło. Dzięki tej metodzie, tło powoli adaptuje się do zmian np. oświetlenia. Proces jest jednak na tyle powolny, że elementy poruszające się pozostawiają bardzo niewielki ślad, nie mający wpływu na metody detekcji.

### Wykrycie ruchu i zmian
Wykrycie ruchomych obiektów odbywa się za pomocą detekcji **MOG2**, natomiast zmian w obrazie za pomocą różnicy jasności obrazów **AbsDiff**: aktualnego oraz tła.

### Progowanie i odszumienie
Oba wyniki są następnie niezależnie progowane w celu wyodrębnienia tylko znaczących elementów, a następnie aplikowane są filtry morfologiczne: **otwarcie**, w celu usunięcia drobnych szumów, a następnie **zamknięcie** w celu połączenia bliskich sobie obiektów.

### Akumulacja zmian
Wykryte zmiany jasności są nakładane jedna po drugiej na osobnym obrazie, z pewnym współczynnikiem i zanikają po pewnym czasie. Obiekty które nie poruszają się, lub robią to bardzo powoli będą zatem jaśniejsze niż te szybkie. 

### Detekcja nieruchomych obiektów
Nagromadzona zmiany jasności są filtrowane przez progowanie. Jeżeli jakiś fragment przekroczył wysoki próg (ok. 250), to oznacza, że w tym miejscu od pewnego czasu znajduje się obiekt nieruchomy (lub prawie nieruchmy).

### Wykrywanie konturów
Jeżeli po progowaniu na wynikowym obrazie znajduje się chociaż 1 obiekt (białe piksele) są na nim wykrywane kontury. Spośród nich, te których powierzchnia przekracza zadaną minimalną powierzchnię wybierany jest największy. Następnie szukany jest odpowiednik na obrazie z połączonych metod MOG oraz AbsDiff. Połączenie takie najlepiej wykrywa ostatnie zmiany w okół danego obiektu: obiekt w całości oraz otoczenie, np. człowieka.

### Przypisanie do _Podejrzanego Obiektu_
Wykryty nieruchomy obiekt oraz jego bliskie otoczenie zakwalifikowane jest jako podejrzane. Każdy podejrzany obiekt ma 3 stany: **inicjalizacja**, **aktywność** oraz **alarm**. Zainicjowany obiekt (kolor biały) tylko mierzy swój czas. Jeżeli nie zostanie wykryty w następnej klatke lub znacznie się poruszy zostanie zresetowany. Jednak w przypadku gdy tak się nie stało i został przekroczony czas aktywacji (domyślnie 3 sekundy) następuje aktywacja. Aktywowany obiekt (kolor zielony) musi pozostać w swoim miejscu do czasu zaalarmowania (domyślnie 5 sekund). Gdy ten zostanie przekroczony obiekt zostaje zgłoszony jako porzucony (kolor czerwony). W rozwojowej wersji faza aktywności służy do wykrywania w pobliżu właściciela, a sam alarm zostanie uruchomiony gdy ten zbytnio oddali się od bagażu.

