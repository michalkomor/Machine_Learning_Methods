
# coding: utf-8

# WCZYTYWANIE I PRZEGLĄDANIE

# In[6]:


#importowanie bibliotek
import pandas as pd
import pandas_profiling as pdp
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas_profiling
import matplotlib.pyplot as plt


# In[7]:


#wyświetlanie wykresów w konsoli Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#definiowanie nazw kolumn
nazwy_kolumn = ["klasa", "mean radius", "mean texture", "mean perimeter", "mean area","mean smoothness", 
                "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", 
                "radius error", "texture error", "perimeter error", "area error", "smoothness error", 
                "compactness error", "concavity error", "concave points error", "symmetry error", 
                "fractal dimension error", "worst radius", "worst texture", "worst perimeter", "worst area", 
                "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", 
                "worst fractal dimension"]


# In[9]:


#wczytanie pliku csv i przypisanie nazw kolumn
nowotwor = pd.read_csv("wdbc.data", names=nazwy_kolumn)


# In[10]:


#pierwsze pięć wierszy
nowotwor.head()


# In[11]:


#zastepowanie nazw klas Ł-łagodny, Z-złośliwy
nowotwor["klasa"] = nowotwor["klasa"].str.replace("B","Ł")
nowotwor["klasa"] = nowotwor["klasa"].str.replace("M","Z")


# In[12]:


#raport opisujący dane tablicy nowotwor
pdp.ProfileReport(nowotwor)


# In[13]:


#okreslenie wielkości tablicy
nowotwor.shape


# In[14]:


#liczba kolumnt
len(nowotwor.columns)


# In[15]:


#przypisanie do tablicy X obserwacji, a do wektora y - klas 
X=nowotwor.drop(["klasa"],axis=1)
y=nowotwor["klasa"]


# In[16]:


#określenie wielkości X i y
X.shape, y.shape


# In[17]:


#utworzenie drzewa dycyzyjnego, kryterium - entropia
drzewo = DecisionTreeClassifier(criterion = "entropy", random_state=42)


# In[18]:


#obiekt trenowany na danych X z klasami y
drzewo.fit(X = X, y = y) 


# In[19]:


#narysowanie drzewa decyzyjnego
dot_data = tree.export_graphviz(drzewo, out_file = None,
                         feature_names = X.columns, class_names = ["Z", "Ł"],
                         filled = True, rounded = True,  
                         special_characters = True)  
graph = graphviz.Source(dot_data)  
graph


# In[20]:


#sprawdzanie dokładności podziału
drzewo.score(X=X,y=y)


# Podział na zbiory train i test 

# In[21]:


#sprawdzenie proporcji klas w zbiorze
y.value_counts(normalize=True)


# In[22]:


#podział na zbiory train i test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)


# In[23]:


#widać, że po po podziale proporcje klas w zbiorach zmieniły się
print(y_train.value_counts(normalize=1))
print(y_test.value_counts(normalize=1))


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42, stratify=y) 
#stratify - wymuszamy ale zostały zachowane proporcje między Ł i Z w zmiennej y
#wartość random_state oznacza od jakiej wartości ma zacząć pracę generator liczb losowych
#który dzieli zbiór y na dwa podzbiory. Jeżeli wpiszemy tu jakąś konkretną wartość 
#to na każdym komputerze zostanie wylosowany taki sam podzbiór


# In[25]:


#teraz po podziale proporcje klas są zbliżone do oryginalnego zbioru
print(y_train.value_counts(normalize=1))
print(y_test.value_counts(normalize=1))


# In[26]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[27]:


X_train.head()


# In[28]:


y_train.head()


# In[29]:


X_test.head()


# In[30]:


y_test.head()


# In[31]:


#zapisanie uzyskanych podzbiorów na dysku:
X_train.to_csv("output/X_train.csv", index_label="index")
X_test.to_csv("output/X_test.csv", index_label="index")
y_train.to_csv("output/y_train.csv", index_label="index")
y_test.to_csv("output/y_test.csv", index_label="index")


# #użycie zbiorów train i test

# In[32]:


#ponownie mozemy wytrenować obiekt drzewo
drzewo.fit(X=X_train,y=y_train)


# In[33]:


#wizualizacja drzewa decyzyjnego
dot_data = tree.export_graphviz(drzewo, out_file = None,
                         feature_names = X.columns, class_names = ["Z", "Ł"],
                         filled = True, rounded = True,  
                         special_characters = True)  
graph = graphviz.Source(dot_data)  
graph


# In[34]:


#teraz do predykcji użyjemy zbioru testowego
predykcja=drzewo.predict(X=X_test)


# In[35]:


wyniki = pd.DataFrame({"predykcja": predykcja, "prawda": y_test})


# In[36]:


wyniki.head()


# In[37]:


wyniki


# In[38]:


#macierz pomyłek
cm= confusion_matrix(y_true = wyniki["prawda"], y_pred=wyniki["predykcja"])
cm


# In[39]:


#wyswietlenie macierzy pomylek na heatmapie
sns.heatmap(cm, annot=True, fmt="" , xticklabels=["Z","Ł"], yticklabels=["Z","Ł"])


# In[40]:


#wysiwetlanie wszystkich nieprawidłowo przewidzianych 
wyniki[(wyniki["prawda"] == "Ł") & (wyniki["predykcja"] == "Z")]


# In[41]:


drzewo.predict_proba(X_test)


# In[42]:


wyniki["prawdopod"] = abs(0.5 - drzewo.predict_proba(X_test)[:,1])


# In[43]:


wyniki[wyniki["predykcja"] != wyniki["prawda"]].sort_values(by = ["prawdopod"])


# In[44]:


drzewo.score(X=X_test, y=y_test)


# In[45]:


#powtarzalnosc budowy drzewa
#https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn


# In[46]:


#HIPERPARAMETRY


# In[47]:


rezultaty=pd.DataFrame()
rezultat= 0
hiperparametry={}

for max_depth in range(1,21):
    drzewo=DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
    drzewo.fit(X=X_train, y=y_train)
    rezultat_testowy = drzewo.score(X=X_test,y=y_test)
    rezultat_treningowy = drzewo.score(X=X_train, y=y_train)
    rezultaty = rezultaty.append({"max_depth": max_depth, 
                                            "rezultat_testowy": rezultat_testowy, 
                                            "rezultat_treningowy": rezultat_treningowy}, ignore_index = True)
    dot_data = tree.export_graphviz(drzewo, out_file = None,
                         feature_names = X.columns, class_names = ["Z", "Ł"],
                         filled = True, rounded = True,  
                         special_characters = True)  
    graph = graphviz.Source(dot_data)  
    graph
    if rezultat_testowy > rezultat:
        print("Mamy lepszy rezultat: {}".format(rezultat_testowy))
        print("max_depth: {}".format(max_depth))
        rezultat = rezultat_testowy
        hiperparametry["max_depth"] = max_depth


# In[48]:


hiperparametry


# In[49]:


rezultaty


# In[50]:


rezultaty.plot(x="max_depth", grid=True)


# In[51]:


drzewo = DecisionTreeClassifier(criterion = "entropy", max_depth = hiperparametry["max_depth"], random_state = 42)
drzewo.fit(X = X_train, y = y_train)


# In[52]:


dot_data = tree.export_graphviz(drzewo, out_file = None,
                         feature_names = X.columns, class_names = ["Z", "Ł"],
                         filled = True, rounded = True,  
                         special_characters = True)  
graph = graphviz.Source(dot_data)  
graph


# In[53]:


#DWA HIPERPARAMETRY


# In[54]:


rezultaty = pd.DataFrame()
rezultat = 0
hiperparametry = {}

for max_depth in range(1,21):
    for min_samples_leaf in range(1, 21):
        drzewo = DecisionTreeClassifier(criterion = "entropy", max_depth = max_depth, 
                                        min_samples_leaf = min_samples_leaf, random_state = 42)
        drzewo.fit(X = X_train, y = y_train)
        rezultat_testowy = drzewo.score(X = X_test, y = y_test)
        rezultat_treningowy = drzewo.score(X = X_train, y = y_train)
        rezultaty = rezultaty.append({"max_depth": max_depth,
                                      "min_samples_leaf": min_samples_leaf,
                                      "rezultat_testowy": rezultat_testowy, 
                                      "rezultat_treningowy": rezultat_treningowy}, ignore_index = True)
        if rezultat_testowy > rezultat:
            print("Mamy lepszy rezultat: {}".format(rezultat_testowy))
            print("max_depth: {}, min_samples_leaf: {}".format(max_depth, min_samples_leaf))
            rezultat = rezultat_testowy
            hiperparametry["max_depth"] = max_depth
            hiperparametry["min_samples_leaf"] = min_samples_leaf


# In[55]:


drzewo = DecisionTreeClassifier(criterion = "entropy", max_depth = hiperparametry["max_depth"], random_state = 42)
drzewo.fit(X = X_train, y = y_train)


# In[56]:


dot_data = tree.export_graphviz(drzewo, out_file = None,
                         feature_names = X.columns, class_names = ["Z", "Ł"],
                         filled = True, rounded = True,  
                         special_characters = True)  
graph = graphviz.Source(dot_data)  
graph


# In[57]:


rezultaty


# In[58]:


macierz = pd.pivot_table(rezultaty[["max_depth", "min_samples_leaf", "rezultat_testowy"]], index = ["max_depth", "min_samples_leaf"]).unstack()


# In[59]:


sns.heatmap(macierz, xticklabels=range(1,21))


# In[60]:


#CZTERY HIPERPARAMETRY


# In[61]:


len(range(1,21)) * len(range(1, 21)) * len(range(2,21)) * len(["entropy", "gini"])
#TYLE DRZEW PRZETESTUJEMY


# In[62]:


rezultaty = pd.DataFrame()
rezultat = 0
hiperparametry = {}

for max_depth in range(1,21):
    print("Testuję max_depth = {}".format(max_depth))
    for min_samples_leaf in range(1, 21):
        for min_samples_split in range(2, 21):
            for criterion in ["entropy", "gini"]:
                drzewo = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth, 
                                                min_samples_leaf = min_samples_leaf,
                                                min_samples_split = min_samples_split, random_state = 42)
                drzewo.fit(X = X_train, y = y_train)
                rezultat_testowy = drzewo.score(X = X_test, y = y_test)
                rezultat_treningowy = drzewo.score(X = X_train, y = y_train)
                rezultaty = rezultaty.append({"max_depth": max_depth,
                                              "min_samples_leaf": min_samples_leaf,
                                              "rezultat_testowy": rezultat_testowy,
                                              "criterion": criterion,
                                              "min_samples_split": min_samples_split,
                                              "rezultat_treningowy": rezultat_treningowy}, ignore_index = True)
                if rezultat_testowy > rezultat:
                    print("Mamy lepszy rezultat: {}".format(rezultat_testowy))
                    print("max_depth: {}, min_samples_leaf: {}, "
                          "min_samples_split: {}, criterion: {}".format(max_depth, min_samples_leaf, min_samples_split,
                                                                        criterion))
                    rezultat = rezultat_testowy
                    hiperparametry["max_depth"] = max_depth
                    hiperparametry["min_samples_leaf"] = min_samples_leaf
                    hiperparametry["min_samples_split"] = min_samples_split
                    hiperparametry["criterion"] = criterion


# In[63]:


#wysiwetlenie uzyskanych hiperparametrów
hiperparametry


# In[64]:


#wytrenowanie drzewa na bazie uzyskanych hiperparametrow
drzewo = DecisionTreeClassifier(criterion = "entropy", max_depth = hiperparametry["max_depth"], 
                                random_state = 42, min_samples_leaf = hiperparametry["min_samples_leaf"],
                                min_samples_split = hiperparametry["min_samples_split"])
drzewo.fit(X = X_train, y = y_train)


# In[65]:


#wyswietlenie drzewa dycyzyjnego
dot_data = tree.export_graphviz(drzewo, out_file = None,
                         feature_names = X.columns, class_names = ["Z", "Ł"],
                         filled = True, rounded = True,  
                         special_characters = True)  
graph = graphviz.Source(dot_data)  
graph


# In[66]:


#predykcja wartości y_test na podstawie danych X_test
predykcja = drzewo.predict(X = X_test)
wyniki = pd.DataFrame({"predykcja": predykcja, "prawda": y_test})
wyniki["prawdopod"] = abs(0.5 - drzewo.predict_proba(X_test)[:,1])
wyniki[wyniki["predykcja"] != wyniki["prawda"]].sort_values(by = ["prawdopod"])


# In[67]:


#GridSearch - automatyczne sprawdzanie kombinacji hiperparametrów 
#CV - CrossValidation - dzieli zbiór treningowy na n różnych części


# In[68]:


from sklearn.model_selection import GridSearchCV


# In[69]:


estimator = DecisionTreeClassifier(random_state=42)


# In[70]:


param_grid = {"max_depth": range(2,9), "min_samples_leaf": range(3,10), "min_samples_split": range(2,17), 
              "criterion": ["entropy", "gini"]}


# In[71]:


param_grid


# In[72]:


#ustawienie klasyfikatora
klasyfikator = GridSearchCV(estimator=estimator, param_grid=param_grid)


# In[74]:


#trenowanie klasyfikatora
get_ipython().run_line_magic('time', '')
klasyfikator.fit(X=X_train, y=y_train)


# In[75]:


#wyswietlenie najlepszych uzyskanych hiperparametrow
hiperparametry=klasyfikator.best_params_
hiperparametry


# In[76]:


#NIESTABILNOŚĆ - czy nasze drzewo jest podatne na to jakie podamy obserwacje, czy mocno zmienią się predykcje


# In[77]:


#import "głupiego klasyfikatora"
from sklearn.dummy import DummyClassifier


# In[78]:


rezultaty = pd.DataFrame()
for ziarno_losowośći in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=ziarno_losowośći, stratify = y)
    drzewo = DecisionTreeClassifier(criterion = hiperparametry["criterion"], max_depth = hiperparametry["max_depth"], 
                                random_state = 42, min_samples_leaf = hiperparametry["min_samples_leaf"],
                                min_samples_split = hiperparametry["min_samples_split"])
    drzewo.fit(X = X_train, y = y_train)
    głupi = DummyClassifier(strategy="stratified", random_state=ziarno_losowośći)
    głupi.fit(X = X_train, y = y_train)
    rezultaty = rezultaty.append({"dokładność_drzewo": drzewo.score(X = X_test, y = y_test), 
                                  "dokładność_głupi": głupi.score(X = X_test, y = y_test)}, ignore_index = True)


# In[79]:


rezultaty


# In[80]:


plt.figure()
rezultaty["dokładność_drzewo"].plot.hist()


# In[81]:


plt.figure()
rezultaty["dokładność_głupi"].plot.hist()


# In[82]:


plt.figure()
rezultaty.plot.hist(alpha=0.5)


# In[83]:


#wyswietlenie raporu rezultatow
pandas_profiling.ProfileReport(rezultaty)

