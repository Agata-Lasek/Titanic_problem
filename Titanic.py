#import numpy as np                 #używane do operacji na macierzach #import keras (niepotrzebne ale zostaw na razie)
import pandas as pd                 #używane do manipulacji danymi
from keras.models import Sequential #umożliwia tworzenie sekwencyjnych modeli sieci neuronowych
from keras.layers import Dense      #warstwy gęste w sieci neuronowej
import random

#warstwa wejsciowa

titanic = pd.read_csv('C:/Users/agata/Desktop/titanic.csv')
test = pd.read_csv('C:/Users/agata/Desktop/test.csv')
        
titanic['Age'] = titanic['Age'].fillna(random.uniform(0, 100))       #aby uzupełnić brakujące dane w kolumnie 'Age', zabezpieczenie, sieci nie rozpoznaja NaN
test['Age'] = test['Age'].fillna(random.uniform(0, 100))             #randomowe liczby dziesietne w przedziale 0-100, zeby nie było pustych

titanic.shape, test.shape, titanic.columns.values                    # informacje o liczbie wierszy, kolumn oraz nazw kolumn dla titanic, test, i połączonej ramy danych df
titanic['WithSb'] = titanic.apply(lambda row: '1' if row['SibSp'] == 1 or row['Parch'] == 1 else 0, axis=1)       # nowa kolumna 'WithSomebody' wykonujaca OR dla SibSp i Parch (jest z kims/sam)
test['WithSb'] = test.apply(lambda row: '1' if row['SibSp'] == 1 or row['Parch'] == 1 else 0, axis=1)      

titanic['WithSb'] = titanic['WithSb'].astype(int)                    #zabezpieczenia przed wczytaniem ciągu znaków, konieczność występywania liczb
test['WithSb'] = test['WithSb'].astype(int)                          #zmiana z ciagow znakow na liczby
titanic['Age'] = titanic['Age'].astype(int)
test['Age'] = test['Age'].astype(int)


print(titanic.shape, test.shape, titanic.columns.values, test.columns.values)     #pomoc wizualna, sprawdza czy kolumny sa odseparowane oraz ilosc wierszy/kolumn ('.shape' inny widok np (891, 13))


# informacje dla warstw ukrytych (wybralam tylko 4 ale mozna poszezyc)

titanic.Sex = titanic.Sex.map({'male':0, 'female':1})                             #map() przyporządkowuje wartości 0 i 1 na podstawie płci, dla uczenia maszynowego wymagane sa numeryczne dane
test.Sex = test.Sex.map({'male': 0, 'female': 1})
selected_columns1 = titanic[['Sex', 'Survived']] 
grouped_by_class1 = selected_columns1.groupby(['Sex'], as_index=False).mean()     #grupuje według unikalnych wartości w wybranej kolumnie i oblicza średnią dla każdej zgrupowanej klasy (sam sprawdza co sie powtarza)
print("\n 0 = MALE   1 = FEMALE \n",grouped_by_class1)                            #pomoc wizualna


selected_columns2 = titanic[['Age', 'Survived']]     
bins = [0, 18, 25, 30, 40,  50, float('inf')]                                     #definiujemy przedziały wiekowe,używane do podziału wartości w 'Age',  float('inf') to przedział, obejmujacy wiek > 50
labels = ['0-18', '18-25', '25-30', '30-40', '40-50', '50+']                      #etykiety do przedziałów wiekowych
selected_columns2['AgeGroup'] = pd.cut(selected_columns2['Age'], bins=bins, labels=labels, right=False)     #cut twoży nowa kolumne "AgeGroup" przypisuje każdemu pasażerowi odpowiedni przedział na podstawie wartości w kolumnie "Age"
grouped_by_class2 = selected_columns2.groupby(['AgeGroup'], as_index=False).mean()                          #grupuje według unikalnych wartości w wybranej kolumnie i oblicza średnią dla każdej zgrupowanej klasy (sam sprawdza co sie powtarza)
print("\n PRZEDZIALY WIEKOWE   \n",grouped_by_class2)                             #pomoc wizualna


selected_columns3 = titanic[['Pclass', 'Survived']]                               #Pclass - klasa pasażerska
grouped_by_class3 = selected_columns3.groupby(['Pclass'], as_index=False).mean()
print("\n 1 = A   2 = B   3 = C \n",grouped_by_class3)


selected_columns4 = titanic[['WithSb', 'Survived']]                               #WithSb - byl sam na pokladzie, byl z kims
grouped_by_class4 = selected_columns4.groupby(['WithSb'], as_index=False).mean()  #grupuje według unikalnych wartości(ograniczone do 0 lub 1)
print("\n 0 = ALONE   1 = TOGETHER   \n",grouped_by_class4)



titanic = titanic.drop(labels=['Name','Cabin','Ticket','Fare','Embarked', 'SibSp', 'Parch'], axis=1) #usuwamy nieuzywane kolumny, axis=1 oznacza, że usuwamy kolumny, nie wiersze
test = test.drop(labels=['Name','Cabin','Ticket','Fare','Embarked','SibSp', 'Parch'], axis=1) #usuwamy nieuzywane kolumny, axis=1 oznacza, że usuwamy kolumny, nie wiersze


# Przygotowanie danych do treningu modelu
X_train = titanic.drop(['Survived'], axis=1)    #zawiera dane wejściowe (cechy), usuwa survived etykiety klasy, które są celem przewidywania
y_train = titanic['Survived'].astype(int)       #zawiera survived które mają być przewidziane

print(titanic.head)                             #pomoc wizualna
print(test.head)

print(X_train.head)
print(y_train.head)

# Inicjalizacja modelu
model = Sequential()                                                                 #Sequential służy do inicjalizacji sieci neuronowej

model.add(Dense(units=8, input_dim=5, activation='relu'))                            #tworze warstwe wejś ciową sieci, input_dim=5, bo tyle daje informacji, (ile wejsc #Dense służy do dodawania warstw do sieci neuronowej. 
model.add(Dense(units=4, activation='relu'))
#model.add(Dense(units=2, activation='relu'))                                         #dwie warstwy ukryte
model.add(Dense(units=1, activation='sigmoid'))                                      # Sigmoid dla problemu binarnej klasyfikacji, warstwa wyjsciowa
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    # Kompilacja modelu, optymalizator adam dostosowuje tempo uczenia się(najbardziej popularny), loss-funkcja straty dla problemów binarnej klasyfikacji. odpowiednia, gdy mamy dwie klasy (przeżył/nie przeżył) 

#print('aaaaaaaaaaaaaaaaaaaaaaaaaaaa',X_train.shape)                                 #sprawdzenie poprawnosci (niech zostanie zakomentowane)
#print('bbbbbbbbbbbbbbbbbbbbbbbbbbb',test.shape)

model.fit(X_train, y_train, batch_size=32, epochs=100)                               # Trenowanie modelu dane treningowe zostaną przetworzone 100 razy przez cały model, batch_size=32 bo najbardziej popularny


# Uzyskanie prognoz na danych testowych

y_pred = model.predict(test)                                                          #do uzyskania prognoz na podstawie danych testowych test
y_final = (y_pred > 0.5).astype(int).reshape(test.shape[0])                           #bedzie zyl jak szansa powyżej 50%, (y_pred > 0.5): Porównuje True lub False .astype(int) zamienia na int, reshape(test.shape[0]): Dostosowuje kształt tablicy wynikowej do kształtu tablicy testowe, 1 linia to 1 linia
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_final})      # Zapisanie wyników do pliku CSV
output['PersonIndex'] = output.index                                                  # pomoc, Dodanie kolumny z numerem indeksu dla osób, które według sieci neuronowej przeżyją
output['SurvivalProbability'] = y_pred.flatten() * 100                                # sprawdzanie poprawności, Procentowa szansa przeżycia, Dodanie kolumny z szansą przeżycia


output.to_csv('C:/Users/agata/Desktop/Sieci_neuronowe/did_they_survived.csv', index=False)

did_they_survived = pd.read_csv('C:/Users/agata/Desktop/Sieci_neuronowe/did_they_survived.csv') # sprawdzanie poprawności
print(did_they_survived)

print("Osoby, które według sieci neuronowej przeżyją:")                                # pomoc, Wypisanie numerów indeksów osób, które według sieci neuronowej przeżyją
print(output[output['Survived'] == 1][['PersonIndex', 'SurvivalProbability']])

survived_count = len(output[output['Survived'] == 1])                                  # sprawdzenie poprawności, Wyświetlenie liczby osób, które przeżyły
print(f"\nLiczba osób, które według sieci neuronowej przeżyły: {survived_count}")