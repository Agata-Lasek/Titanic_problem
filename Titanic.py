import numpy as np          #używane do operacji na macierzach
import pandas as pd         #używane do manipulacji danymi
import keras
from keras.models import Sequential
from keras.layers import Dense

#warstwa wejsciowa

titanic = pd.read_csv('C:/Users/agata/Desktop/titanic.csv')
test = pd.read_csv('C:/Users/agata/Desktop/test.csv')

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())         #aby uzupełnić brakujące dane w kolumnie 'Age'
test['Age'] = test['Age'].fillna(titanic['Age'].median())

#df = titanic.append(test , ignore_index = True)        #laczy 2 ramy titanic i test i ignoruje indeksy obu ramek danych i przypisuje nowe indeksy od zera
titanic.shape, test.shape, titanic.columns.values       # informacje o liczbie wierszy, kolumn oraz nazw kolumn dla titanic, test, i połączonej ramy danych df
titanic['WithSb'] = titanic.apply(lambda row: '1' if row['SibSp'] == 1 or row['Parch'] == 1 else 0, axis=1)       # nowa kolumna 'WithSomebody' wykonujaca OR dla SibSp i Parch (jest z kims/sam)
test['WithSb'] = test.apply(lambda row: '1' if row['SibSp'] == 1 or row['Parch'] == 1 else 0, axis=1)      

#zmiana z ciagow znakow na liczby
titanic['WithSb'] = titanic['WithSb'].astype(int)
test['WithSb'] = test['WithSb'].astype(int)


print(titanic.shape, test.shape, titanic.columns.values, test.columns.values)    #sprawdza czy kolumny sa odseparowane oraz ilosc wierszy/kolumn ('.shape' inny widok np (891, 13))


#warstwy ukryte

titanic.Sex = titanic.Sex.map({'male':0, 'female':1})                            #map() przyporządkowuje wartości 0 i 1 na podstawie płci, dla uczenia maszynowego wymagane sa numeryczne dane
test.Sex = test.Sex.map({'male': 0, 'female': 1})
selected_columns1 = titanic[['Sex', 'Survived']] 
grouped_by_class1 = selected_columns1.groupby(['Sex'], as_index=False).mean()    #grupuje według unikalnych wartości w wybranej kolumnie i oblicza średnią dla każdej zgrupowanej klasy (sam sprawdza co sie powtarza)
print("\n 0 = MALE   1 = FEMALE \n",grouped_by_class1)


selected_columns2 = titanic[['Age', 'Survived']]     
bins = [0, 18, 25, 30, 40,  50, float('inf')]                                 #definiujemy przedziały wiekowe,używane do podziału wartości w 'Age',  float('inf') to przedział, obejmujacy wiek > 50
labels = ['0-18', '18-25', '25-30', '30-40', '40-50', '50+']                  #etykiety do przedziałów wiekowych
selected_columns2['AgeGroup'] = pd.cut(selected_columns2['Age'], bins=bins, labels=labels, right=False)     #cut twoży nowa kolumne "AgeGroup" przypisuje każdemu pasażerowi odpowiedni przedział na podstawie wartości w kolumnie "Age"
grouped_by_class2 = selected_columns2.groupby(['AgeGroup'], as_index=False).mean()                          #grupuje według unikalnych wartości w wybranej kolumnie i oblicza średnią dla każdej zgrupowanej klasy (sam sprawdza co sie powtarza)
print("\n PRZEDZIALY WIEKOWE   \n",grouped_by_class2)


selected_columns3 = titanic[['Pclass', 'Survived']]                               #Pclass - klasa pasażerska
grouped_by_class3 = selected_columns3.groupby(['Pclass'], as_index=False).mean()
print("\n 1 = A   2 = B   3 = C \n",grouped_by_class3)


selected_columns4 = titanic[['WithSb', 'Survived']]                               #WithSb - byl sam na pokladzie, byl z kims
grouped_by_class4 = selected_columns4.groupby(['WithSb'], as_index=False).mean()  #!!!!!!!!musze zmienic wczytywanie gdy ktos jest z kilkoma osobami SibSp  Parch >1
#print("\n 0 = ALONE   1 = TOGETHER   \n",grouped_by_class4)


#warstwa wyjsciowa

titanic = titanic.drop(labels=['Name','Cabin','Ticket','Fare','Embarked', 'SibSp', 'Parch'], axis=1) #usuwamy nieuzywane kolumny, axis=1 oznacza, że usuwamy kolumny, nie wiersze
test = test.drop(labels=['Name','Cabin','Ticket','Fare','Embarked','SibSp', 'Parch'], axis=1) #usuwamy nieuzywane kolumny, axis=1 oznacza, że usuwamy kolumny, nie wiersze

# Przygotowanie danych do modelu
X_train = titanic.drop(['Survived'], axis=1)
y_train = titanic['Survived'].astype(int)

print(titanic.head)
print(test.head)

print(X_train.head)
print(y_train.head)

# Inicjalizacja modelu
model = Sequential()                                                                    #Sequential służy do inicjalizacji sieci neuronowej

model.add(Dense(units=5, input_dim=5, activation='relu'))                             #tworze I warstwe sieci (ile wejsc #Dense służy do dodawania warstw do sieci neuronowej.
model.add(Dense(units=3, activation='relu'))                                            #dwie warstwy ukryte
model.add(Dense(units=3, activation='relu'))  
model.add(Dense(units=1, activation='sigmoid'))                                         # Sigmoid dla problemu binarnej klasyfikacji, warstwa wyjsciowa
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])       # Kompilacja modelu

#print('aaaaaaaaaaaaaaaaaaaaaaaaaaaa',X_train.shape)  #sprawdzenie poprawnosci
#print('bbbbbbbbbbbbbbbbbbbbbbbbbbb',test.shape)

model.fit(X_train, y_train, batch_size=32, epochs=50)                # Trenowanie modelu     #Trenowanie, czyli dane treningowe zostaną przetworzone 50 razy przez cały model


#test jak sobie radzi siec  i zapisania ich do pliku CSV
# Uzyskanie prognoz na danych testowych

y_pred = model.predict(test)                                                            #do uzyskania prognoz na podstawie danych testowych test
y_final = (y_pred > 0.5).astype(int).reshape(test.shape[0])                             #bedzie zyl jak szansa powyżej 50%
output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_final})        # Zapisanie wyników do pliku CSV
output.to_csv('did_they_survived.csv', index=False)


did_they_survived = pd.read_csv('C:/Users/agata/Desktop/Sieci_neuronowe/did_they_survived.csv')
print(did_they_survived.head(30))
