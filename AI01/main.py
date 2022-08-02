"""
PROBLEMA 1
Să se determine ultimul (din punct de vedere alfabetic) cuvânt care poate apărea într-un text care conține
mai multe cuvinte separate prin ” ” (spațiu).
De ex. ultimul (dpdv alfabetic) cuvânt din ”Ana are mere rosii si galbene” este cuvântul "si".

Descrierea solutiei:
IN: prop - string, propozitia in care se cauta ultimul cuvant dpdv alfabetic
OUT: last_word - string, cuvantul cautat
Functia cauta ultimul cuvant dpdv alfabetic, iterand prin cuvintele propozitiei initiale memorate in variabila
"text" in urma separarii cuvintelor prin spatiu si verificand daca ultimul cuvant curent este dupa cuvantul iterat
in momentul acela.
"""
import heapq
import math
import time
from queue import Queue


def solve_1(prop):
    text = prop.split(" ")
    last_word = text[0]
    for word in text[1:]:
        if last_word.lower() < word.lower():
            last_word = word
    return last_word


def test_solve_1():
    assert (solve_1("Ana are mere rosii si galbene") == "si")
    assert (solve_1("Ana Are Mere Rosii Si Galbene") == "Si")
    assert (solve_1("ana are mere rosii si galbene") == "si")
    assert (solve_1("Ana Are Mere Rosii si Galbene") == "si")
    assert (solve_1("ana are mere rosii Si galbene") == "Si")
    assert (solve_1("Ana are mere rosii si SI sI Si galbene") == "si")
    assert (solve_1("") == "")


"""
PROBLEMA 2
Să se determine distanța Euclideană între două locații identificate prin perechi de numere. 
De ex. distanța între (1,5) și (4,1) este 5.0.

Descrierea solutiei:
IN: pair1, pair2 - lista, cele doua perechi de numere pentru care se calculeaza distanta euclidiana
OUT: intreg, distanta euclidiana calculata
Functia memoreaza in lista "distance" patratul diferentei dintre fiecare pereche de elemente (a, b) din cele doua
perechi, apoi calculeaza si returneaza radacina sumei elementelor din "distance", reprezentand distanta euclidiana 
dintre cele doua locatii indicate de cele doua liste, "pair1" si "pair2".
"""


def solve_2(pair1, pair2):
    distance = [(a - b) ** 2 for a, b in zip(pair1, pair2)]
    return math.sqrt(sum(distance))


"""
Descrierea solutiei:
IN: pair1, pair2 - lista, cele doua perechi de numere pentru care se calculeaza distanta euclidiana
OUT: intreg, distanta euclidiana calculata
Functia memoreaza in lista "distances" patratul diferentei dintre fiecare pereche de elemente (a, b) din cele
doua perechi, apoi itereaza aceasta lista si calculeaza in "sum" suma distantelor, returnand radacina acesteia,
reprezentand distanta euclidiana dintre cele doua locatii indicate de cele doua liste, "pair1" si "pair2".
"""


def solve_2_a(pair1, pair2):
    distances = [(pair1[i] - pair2[i]) ** 2 for i in range(len(pair1))]
    sum_of_distances = 0
    for distance in distances:
        sum_of_distances += distance
    return math.sqrt(sum_of_distances)


def test_solve_2():
    assert (abs(solve_2((1, 5), (4, 1)) - 5) < 0.0000001)
    assert (abs(solve_2((5, 1), (1, 4)) - 5) < 0.0000001)
    assert (abs(solve_2((4, 1), (1, 5)) - 5) < 0.0000001)
    assert (abs(solve_2((3, 2, 0), (4, 1, math.sqrt(2))) - 2) < 0.0000001)
    assert (abs(solve_2((1, 5), (1, 5))) < 0.0000001)
    assert (abs(solve_2_a((1, 5), (4, 1)) - 5) < 0.0000001)
    assert (abs(solve_2_a((5, 1), (1, 4)) - 5) < 0.0000001)
    assert (abs(solve_2_a((4, 1), (1, 5)) - 5) < 0.0000001)
    assert (abs(solve_2_a((3, 2, 0), (4, 1, math.sqrt(2))) - 2) < 0.0000001)
    assert (abs(solve_2_a((1, 5), (1, 5))) < 0.0000001)


"""
PROBLEMA 3
Să se determine produsul scalar a doi vectori rari care conțin numere reale.
Un vector este rar atunci când conține multe elemente nule. 
Vectorii pot avea oricâte dimensiuni. De ex. produsul scalar a 2 
vectori unisimensionali [1,0,2,0,3] și [1,2,0,3,1] este 4.

Descrierea solutiei:
IN: values1, values2 - liste, cei doi vectori rari
OUT: product - real, produsul celor doi vectori rari
Functia cauta perechile de elemente care se afla pe aceeasi pozitie in cei doi vectori
si adauga produsul acestora la produsul scalar in cazul in care ambele elemente sunt nenule.
"""


def solve_3(values1, values2):
    product = 0
    for position in range(len(values1)):
        if values1[position] != 0 and values2[position] != 0:
            product += values1[position] * values2[position]
    return product


def test_solve_3():
    assert (solve_3([1, 0, 2, 0, 3], [1, 2, 0, 3, 1]) == 4)
    assert (solve_3([0, 0, 2, 0, 3], [1, 2, 0, 3, 0]) == 0)
    assert (solve_3([1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 4], [1, 2, 0, 3, 1, 1, 1, 1, 0, 0, 1]) == 8)
    assert (solve_3([1, 0, 2, 0, 3, 0, 0, 1, 0, 0, 4], [1, 2, 0, 3, 0, 0, 1, 1, 0, 0, 0]) == 2)
    assert (solve_3([1, 0, 0, 2, 3], [math.sqrt(2), 2, 0, 0.75, 0]) - (math.sqrt(2) + 1.5) < 0.0000000001)


"""
PROBLEMA 4
Să se determine cuvintele unui text care apar exact o singură dată în acel text. 
De ex. cuvintele care apar o singură dată în ”ana are ana are mere rosii ana" sunt: 'mere' și 'rosii'.

Descrierea solutiei:
IN: prop - string, propozitia in care se cauta cuvintele care nu se repeta
OUT - words - lista de string-uri, cuvinteole care nu se repeta
Functia creeaza un dictionar in care retine cuvintele care apar in propozitie si frecventa acestora, 
apoi memoreaza in lista "words" cuvintele care apar o singura data si le returneaza.
"""


def solve_4(prop):
    dictionary = {}
    for word in prop.split(" "):
        if word in dictionary.keys():
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    words = []
    for key in dictionary:
        if dictionary[key] == 1:
            words.append(key)
    return words


def test_solve_4():
    assert (solve_4("ana are ana are mere rosii ana") == ["mere", "rosii"])
    assert (solve_4("ana are mere rosii") == ["ana", "are", "mere", "rosii"])
    assert (solve_4("ana ana") == [])
    assert (solve_4("Ana ANA ana aNA aNa") == ["Ana", "ANA", "ana", "aNA", "aNa"])


"""
PROBLEMA 5
Pentru un șir cu n elemente care conține valori din mulțimea {1, 2, ..., n - 1} astfel încât o singură valoare se repetă
de două ori, să se identifice acea valoare care se repetă. De ex. în șirul [1,2,3,4,2] valoarea 2 apare de două ori.

Descrierea solutiei:
IN: values - lista, sirul de numere dat
OUT: intreg, numarul care se repeta
Functia calculeaza suma numerelor din lista "values", apoi scade din aceasta suma numerelor de la 1 la n - 1, 
unde n reprezinta lungimea listei "values", obtinand astfel elementul care se repeta.
"""


def solve_5(values):
    my_sum = sum(values)
    expected_sum = 0
    for i in range(1, len(values)):
        expected_sum += i
    return my_sum - expected_sum


def test_solve_5():
    assert (solve_5([1, 2, 3, 4, 2]) == 2)
    assert (solve_5([1, 2, 3, 4, 5, 6, 6, 7, 8]) == 6)
    assert (solve_5([]) == 0)
    assert (solve_5([1, 1]) == 1)


"""
PROBLEMA 6
Pentru un șir cu n numere întregi care conține și duplicate, 
să se determine elementul majoritar (care apare de mai mult de n / 2 ori). 
De ex. 2 este elementul majoritar în șirul [2,8,7,2,2,5,2,3,1,2,2].

Descrierea solutiei:
IN: values - lista, sirul cu n numere cu duplicate in care se cauta elementul majoritar
OUT: intreg, elementul majoritar, daca acesta exista
     altfel, None
Functia creeaza un dictionar in care retine fiecare element din sir si frecventa acestuia,
apoi verifica daca una dintre chei (elementele din sirul initial) respecta conditia de element majoritar,
adica are frecventa mai mare decat n/2. Daca exista, il returneaza, iar daca nu exista, returneaza None.
"""


def solve_6(values):
    if not values:
        return None
    dictionary = {}
    for value in values:
        if value in dictionary.keys():
            dictionary[value] += 1
        else:
            dictionary[value] = 1
    for key in dictionary:
        if dictionary[key] > (len(values) / 2):
            return key
    return None


"""
IN: values - lista, sirul cu n numere cu duplicate in care se cauta elementul majoritar
OUT: intreg, elementul majoritar, daca acesta exista
     altfel, None
Functia cauta un candidat pentru elementul majoritar, folosindu-ne de ideea ca elementul majoritar
apare in sir de mai mult de n / 2 ori, asadar putem observa care element apare cel mai des iterand o singura data sirul.
(Un element care ajunge sa aiba frecventa 0 nu poate fi element majoritar.)
Dupa ce este ales candidatul, se trece iar prin fiecare element al sigurul si se calculeaza frecventa acestuia,
apoi se verifica ca aceasta sa fie mai mare decat n / 2, candidatul fiind chiar element majoritar si returnandu-se.
Daca nu este respectata conditia, inseamna ca nu exista element majoritar si se returneaza None.
"""


def solve_6_a(values):
    if not values:
        return None
    majority = values[0]
    count = 0
    for value in values:
        if value == majority:
            count += 1
        else:
            count -= 1
        if count <= 0:
            majority = value
            count = 1
    count = 0
    for value in values:
        if value == majority:
            count += 1
        if count > (len(values) / 2):
            return value
    return None


def test_solve_6():
    assert (solve_6([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]) == 2)
    assert (solve_6_a([2, 8, 7, 2, 2, 5, 2, 3, 1, 2, 2]) == 2)
    assert (solve_6([6, 5, 6, 6, 7, 6, 8, 6, 9]) == 6)
    assert (solve_6_a([6, 5, 6, 6, 7, 6, 8, 6, 9]) == 6)
    assert (solve_6([6, 7, 8, 9, 9, 10]) is None)
    assert (solve_6_a([6, 7, 8, 9, 9, 10]) is None)
    assert (solve_6([]) is None)
    assert (solve_6_a([]) is None)
    assert (solve_6([1, 2, 1, 2]) is None)
    assert (solve_6_a([1, 2, 1, 2]) is None)


"""
PROBLEMA 7
Să se determine al k-lea cel mai mare element al unui șir de numere cu n elemente (k < n). 
De ex. al 2-lea cel mai mare element din șirul [7,4,6,3,9,1] este 7.

Descrierea solutiei:
IN: values - lista, sirul de numere
    k - intreg, al catelea cel mai mare element este cautat
OUT: intreg, al k-lea cel mai mare element
Functia creeaza un min heap in care retine opusele numerelor din lista "values",
apoi scote din heap primele k-1 elemente (cele mai mari deoarece in radacina va fi mereu "-cel mai mare element")
si returneaza, in final, al k-lea cel mai mare element.
"""


def solve_7(values, k):
    min_heap = []
    for i in range(len(values)):
        heapq.heappush(min_heap, -1 * values[i])
    while k > 1:
        k -= 1
        heapq.heappop(min_heap)
    return -1 * min_heap[0]


def test_solve_7():
    assert (solve_7([7, 4, 6, 3, 9, 1], 2) == 7)
    assert (solve_7([7, 4, 6, 3, 9, 1], 3) == 6)
    assert (solve_7([7, 4, 6, 3, 9, 1], 4) == 4)
    assert (solve_7([7, 4, 6, 3, 9, 7], 2) == 7)
    assert (solve_7([3, 1], 2) == 1)
    assert (solve_7([7, 7, 7, 7], 2) == 7)


"""
PROBLEMA 8
Să se genereze toate numerele (în reprezentare binară) cuprinse între 1 și n. 
De ex. dacă n = 4, numerele sunt: 1, 10, 11, 100.

Descrierea solutiei:
IN: n - intreg, numarul pana la care se genereaza numerele binare
OUT: numbers - lista, lista cu numerele binare de la 1 la n
Functia itereaza numerele de la 1 la n si memoreaza in variabila "binary" 
transformarea fiecarui numar din baza 10 in baza 2 prin impartiri repetate,
valoarea din variabila fiind apoi stocata in lista "numbers" care va fi returnata la final.
"""


def solve_8(n):
    numbers = []
    for number in range(1, n + 1):
        binary = 0
        position = 1
        while number > 0:
            binary += (number % 2) * position
            number = number // 2
            position *= 10
        numbers.append(binary)
    return numbers


"""
Descrierea solutiei:
IN: n - intreg, numarul pana la care se genereaza numerele binare
OUT: numbers - lista, lista cu numerele binare de la 1 la n
Functia foloseste o coada pentru a obtine toate numerele pana la n inclusiv in baza 2,
punand dupa numarul curent scos din coada mai intai 0, apoi 1, pentru a obtine urmatoarele numere
si le memoreaza in lista "numbers" care este apoi returnata.
"""


def solve_8_a(n):
    numbers = []
    queue = Queue()
    queue.put("1")
    for i in range(n):
        current = queue.get()
        numbers.append(int(current))
        queue.put(current + "0")
        queue.put(current + "1")
    return numbers


def test_solve_8():
    assert (solve_8(4) == [1, 10, 11, 100])
    assert (solve_8(1) == [1])
    assert (solve_8(0) == [])
    assert (solve_8(10) == [1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010])
    assert (solve_8_a(4) == [1, 10, 11, 100])
    assert (solve_8_a(1) == [1])
    assert (solve_8_a(0) == [])
    assert (solve_8_a(10) == [1, 10, 11, 100, 101, 110, 111, 1000, 1001, 1010])


"""
PROBLEMA 9
Considerându-se o matrice cu n x m elemente întregi și o listă cu perechi formate din coordonatelea 2 căsuțe
din matrice ((p,q) și (r,s)), să se calculeze suma elementelor din sub-matricile identificate de fieare pereche.
De ex, pt matricea:
[[0, 2, 5, 4, 1],
 [4, 8, 2, 3, 7],
 [6, 3, 4, 6, 2],
 [7, 3, 1, 8, 3],
 [1, 5, 7, 9, 4]]
și lista de perechi ((1, 1) și (3, 3)), ((2, 2) și (4, 4)), suma elementelor din prima sub-matrice este 38, 
iar suma elementelor din a 2-a sub-matrice este 44.

Descrierea solutiei:
IN: matrix - matrice, lista de liste
    pair1 - tuple de tuple-uri, prima pereche
    pair2 - tuple de tuple-uri, a doua pereche
OUT: tuple de intregi, cele doua sume ale elementelor sub-matricelor
Functia calculeaza in "matrix" matricea sumelor si folosindu-se de aceasta, calculeaza suma sub-matricelor 
determinate de punctele date de pair1 si pair2, apleand functia "calculate". (Ceea ce se intampla este faptul 
ca se taie ceea ce este deasupra si in stanga primului punct al unei perechi si se ia tot pana la al doilea punct al
fiecarei perechi.
In cazul in care sub-matricea identificata de puncte este invalida, suma returnata pentru aceasta este valoarea None.
"""


def solve_9(matrix, pair1, pair2):
    def calculate(x0, y0, x1, y1):
        if x0 > x1 or y0 > y1:
            return None
        if x0 == 0 and y0 == 0:
            return matrix[x1][y1]
        if x0 == 0:
            return matrix[x1][y1] - matrix[x1][y0 - 1]
        if y0 == 0:
            return matrix[x1][y1] - matrix[x0 - 1][y1]
        return matrix[x1][y1] - matrix[x0 - 1][y1] - matrix[x1][y0 - 1] + matrix[x0 - 1][y0 - 1]

    px = pair1[0][0]
    py = pair1[0][1]
    qx = pair1[1][0]
    qy = pair1[1][1]
    rx = pair2[0][0]
    ry = pair2[0][1]
    sx = pair2[1][0]
    sy = pair2[1][1]
    for column in range(1, len(matrix[0])):
        matrix[0][column] += matrix[0][column - 1]
    for row in range(1, len(matrix)):
        matrix[row][0] += matrix[row - 1][0]
    for row in range(1, len(matrix)):
        for column in range(1, len(matrix[0])):
            matrix[row][column] += matrix[row - 1][column] + matrix[row][column - 1] - matrix[row - 1][column - 1]
    return calculate(px, py, qx, qy), calculate(rx, ry, sx, sy)


def test_solve_9():
    assert (solve_9([[0, 2, 5, 4, 1],
                     [4, 8, 2, 3, 7],
                     [6, 3, 4, 6, 2],
                     [7, 3, 1, 8, 3],
                     [1, 5, 7, 9, 4]], ((1, 1), (3, 3)), ((2, 2), (4, 4))) == (38, 44))
    assert (solve_9([[0, 2, 5, 4, 1],
                     [4, 8, 2, 3, 7],
                     [6, 3, 4, 6, 2],
                     [7, 3, 1, 8, 3],
                     [1, 5, 7, 9, 4]], ((1, 2), (3, 4)), ((2, 1), (4, 2))) == (36, 23))
    assert (solve_9([[0, 2, 5, 4, 1],
                     [4, 8, 2, 3, 7],
                     [6, 3, 4, 6, 2],
                     [7, 3, 1, 8, 3],
                     [1, 5, 7, 9, 4]], ((4, 1), (3, 4)), ((3, 2), (4, 1))) == (None, None))
    assert (solve_9([[0, 2, 5, 4, 1],
                     [4, 8, 2, 3, 7],
                     [6, 3, 4, 6, 2],
                     [7, 3, 1, 8, 3],
                     [1, 5, 7, 9, 4]], ((0, 0), (1, 2)), ((1, 0), (2, 1))) == (21, 21))
    assert (solve_9([[0, 2, 5, 4, 1],
                     [4, 8, 2, 3, 7],
                     [6, 3, 4, 6, 2],
                     [7, 3, 1, 8, 3],
                     [1, 5, 7, 9, 4]], ((1, 1), (1, 2)), ((0, 2), (2, 2))) == (10, 11))


"""
PROBLEMA 10
Considerându-se o matrice cu n x m elemente binare (0 sau 1) sortate crescător pe linii,
să se identifice indexul liniei care conține cele mai multe elemente de 1.
De ex. în matricea:
[[0,0,0,1,1],
 [0,1,1,1,1],
 [0,0,1,1,1]]
a doua linie conține cele mai multe elemente 1.

Descrierea solutiei:
IN: matrix - lista de liste, matricea in care se cauta indexul liniei cu cele mai multe
elemente de 1
OUT: searched_row - intreg, linia cautata
Functia calculeaza pozitia primului element 1 de pe fiecare linie a matricei apeland functia "search", care
are la baza un algoritm de cautare binara, apoi returneaza index-ul liniei cu pozitia minima, daca aceasta exista,
altfel returneaza None.
"""


def solve_10(matrix):
    def search(left, right, current_row):
        if matrix[current_row][0] == 1:
            return 0
        if left < right and matrix[current_row][-1] != 0:
            middle = (left + right) // 2
            if matrix[current_row][middle] == 1 and matrix[current_row][middle - 1] == 0:
                return middle
            elif matrix[current_row][middle] == 0 and matrix[current_row][middle + 1] == 1:
                return middle + 1
            elif matrix[current_row][middle] == 1:
                return search(left, middle - 1, current_row)
            else:
                return search(middle + 1, right, current_row)
        return len(matrix[0]) + 1

    position = len(matrix[0])
    searched_row = -1
    length = len(matrix[0])
    for row in range(0, len(matrix)):
        current_position = search(0, length, row)
        if position >= current_position:
            position = current_position
            searched_row = row
    if searched_row != -1:
        return searched_row
    return None


def test_solve_10():
    assert (solve_10([[0, 0, 0, 1, 1],
                      [0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1]]) == 1)
    assert (solve_10([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]) is None)
    assert (solve_10([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0]]) == 5)
    assert (solve_10([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]]) == 6)
    assert (solve_10([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1]]) == 5)


"""
PROBLEMA 11
Considerându-se o matrice cu n x m elemente binare (0 sau 1), 
să se înlocuiască cu 1 toate aparițiile elementelor egale cu 0 care sunt complet înconjurate de 1.
De ex. matricea:
[[1,1,1,1,0,0,1,1,0,1],
 [1,0,0,1,1,0,1,1,1,1],
 [1,0,0,1,1,1,1,1,1,1],
 [1,1,1,1,0,0,1,1,0,1],
 [1,0,0,1,1,0,1,1,0,0],
 [1,1,0,1,1,0,0,1,0,1],
 [1,1,1,0,1,0,1,0,0,1],
 [1,1,1,0,1,1,1,1,1,1]]
*devine *
[[1,1,1,1,0,0,1,1,0,1],
 [1,1,1,1,1,0,1,1,1,1],
 [1,1,1,1,1,1,1,1,1,1],
 [1,1,1,1,1,1,1,1,0,1],
 [1,1,1,1,1,1,1,1,0,0],
 [1,1,1,1,1,1,1,1,0,1],
 [1,1,1,0,1,1,1,0,0,1],
 [1,1,1,0,1,1,1,1,1,1]].
 
Descrierea solutiei:
IN: matrix - lista de liste
OUT: matrix - lista de liste, matricea initiala dupa ce au fost inlocuite 0-urile inconjurate de 1 cu 1
Functia inlocuieste cu vlaorea -1 toate 0-urile din matrice, apoi parcurge rama matricei si cand gasteste o
valoare -1 apeleaza functia "flood_fill" pentru pozitia curenta, transformand elementul curent in 0 si apoi 
se apeleaza recursiv pentru toti vecinii lui. In final, inlocuieste valorile -1 ramase in matrice cu 1,
insemnand ca acele elemente sunt inconjurate conplet de valori de 1.
"""


def solve_11(matrix):
    def flood_fill(current_row, current_col):
        if current_row < 0 or current_row >= n or current_col < 0 \
                or current_col >= m or matrix[current_row][current_col] != -1:
            return
        matrix[current_row][current_col] = 0
        flood_fill(current_row + 1, current_col)
        flood_fill(current_row - 1, current_col)
        flood_fill(current_row, current_col + 1)
        flood_fill(current_row, current_col - 1)

    n = len(matrix)
    m = len(matrix[0])
    for row in range(n):
        for col in range(m):
            if matrix[row][col] == 0:
                matrix[row][col] = -1
    for row in range(n):
        if matrix[row][0] == -1:
            flood_fill(row, 0)
    for row in range(n):
        if matrix[row][m - 1] == -1:
            flood_fill(row, m - 1)
    for col in range(m):
        if matrix[0][col] == -1:
            flood_fill(0, col)
    for col in range(m):
        if matrix[n - 1][col] == -1:
            flood_fill(n - 1, col)
    for row in range(n):
        for col in range(m):
            if matrix[row][col] == -1:
                matrix[row][col] = 1
    return matrix


def test_solve_11():
    assert (solve_11([[1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                      [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                      [1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                      [1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                      [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]]) == [[1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                                                           [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                                                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                                           [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                                                           [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]])
    assert (solve_11([[1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                      [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                      [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                      [1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                      [1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                      [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]]) == [[1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                                                           [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                                                           [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                                           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                                                           [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                                                           [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]])


if __name__ == "__main__":
    tic = time.perf_counter()
    test_solve_1()
    test_solve_2()
    test_solve_3()
    test_solve_4()
    test_solve_5()
    test_solve_6()  # 2 metode
    test_solve_7()
    test_solve_8()  # 2 metode
    test_solve_9()
    test_solve_10()
    test_solve_11()
    toc = time.perf_counter()
    print("All tests have been successfully completed in " + str(toc - tic) + " sec.")
