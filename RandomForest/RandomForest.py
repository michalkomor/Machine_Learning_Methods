
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas_profiling

get_ipython().run_line_magic('matplotlib', 'inline')
#wyswietlanie wykresow. 


# In[2]:


X_train = pd.read_csv("output/X_train.csv", index_col = "index")
y_train = pd.read_csv("output/y_train.csv", names = ["index", "klasa"], index_col = "index")

X_test = pd.read_csv("output/X_test.csv", index_col = "index")
y_test = pd.read_csv("output/y_test.csv", names = ["index", "klasa"], index_col = "index")


# In[3]:


#dokladamy asercje, czyli zwraca błąd jeżeli w tym miejscu jest wartość false. TAki "hamulec bezpieczeństwa"
assert (X_train.index == y_train.index).all() == True
assert (X_test.index == y_test.index).all() == True


# In[4]:


#Uruchomienie prostego lasu losowego
#funkcja budująca las losowy


# In[5]:


from sklearn.ensemble import RandomForestClassifier


# In[6]:


las=RandomForestClassifier(criterion="entropy", random_state=42, bootstrap=False, max_features=None)  
#bootstrap=False - bazujemy na pełnym zbiorze


# In[7]:


las.fit(X=X_train, y = y_train.values.ravel())  #ravel - odwrócenie z kolumny na wiersz. Wykonujemy trening


# In[10]:


las.score(X=X_test, y=y_test)


# In[11]:


las.estimators_


# In[12]:


#LAS NA BAZIE POPRZEDNIO WYLICZONYCH HIPERPARAMETRÓW


# In[13]:


hiperparametry = {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 5, 'min_samples_split': 10}


# In[14]:


las = RandomForestClassifier(criterion = "entropy", random_state = 42, bootstrap = False, max_features = None, 
                             max_depth = hiperparametry["max_depth"], 
                             min_samples_leaf = hiperparametry["min_samples_leaf"], 
                             min_samples_split = hiperparametry["min_samples_split"])


# In[15]:


las.fit(X = X_train, y = y_train.values.ravel())


# In[16]:


las.score(X = X_test, y = y_test)


# In[17]:


las.estimators_


# In[18]:


#lasy utworzone do tej pory nie były za bardzo losowe , bo bazowały na tych samych danych i parametrach
#teraz użyjemy parametru bootstrap=true, aby wykorzystać losowanie ze zwracaniem


# In[19]:


las = RandomForestClassifier(criterion = "entropy", random_state = 42, bootstrap = True, max_features = None, 
                             max_depth = hiperparametry["max_depth"], 
                             min_samples_leaf = hiperparametry["min_samples_leaf"], 
                             min_samples_split = hiperparametry["min_samples_split"])


# In[20]:


las.fit(X = X_train, y = y_train.values.ravel())


# In[21]:


las.score(X=X_test, y=y_test)


# In[22]:


#selekcja cech - użyjemy sugerowanej metody - pierwiastek ze wszystkich cech
#max_features=sqrt(n_features)


# In[23]:


X_train.columns


# In[24]:


max_features = int(np.sqrt(len(X_train.columns))) #pierwiastek z 30 int zamienia liczbę zmiennoprzecinkową na całkowitą


# In[25]:


max_features


# In[26]:


#teraz będzie brane pod uwagę 5 losowych cech
las = RandomForestClassifier(criterion = "entropy", random_state = 42, bootstrap = True, max_features = max_features, 
                             max_depth = hiperparametry["max_depth"], 
                             min_samples_leaf = hiperparametry["min_samples_leaf"], 
                             min_samples_split = hiperparametry["min_samples_split"])


# In[27]:


las.fit(X = X_train, y = y_train.values.ravel())
las.score(X = X_test, y = y_test)


# In[28]:


#TEST STABILNOŚCI


# In[29]:


nazwy_kolumn = ["klasa", "mean radius", "mean texture", "mean perimeter", "mean area","mean smoothness", 
                "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension", 
                "radius error", "texture error", "perimeter error", "area error", "smoothness error", 
                "compactness error", "concavity error", "concave points error", "symmetry error", 
                "fractal dimension error", "worst radius", "worst texture", "worst perimeter", "worst area", 
                "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", 
                "worst fractal dimension"]

nowotwor = pd.read_csv("wdbc.data", names = nazwy_kolumn)

nowotwor["klasa"] = nowotwor["klasa"].str.replace("B", "Ł")
nowotwor["klasa"] = nowotwor["klasa"].str.replace("M", "Z")

X = nowotwor.drop(["klasa"], axis = 1)
y = nowotwor["klasa"]

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split


# In[30]:


get_ipython().run_cell_magic('time', '', 'rezultaty = pd.DataFrame()\nfor ziarno_losowośći in range(0,100):\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=ziarno_losowośći, stratify = y)\n    las = RandomForestClassifier(criterion = "entropy", random_state = 42, bootstrap = True, max_features = max_features, \n                             max_depth = hiperparametry["max_depth"], \n                             min_samples_leaf = hiperparametry["min_samples_leaf"], \n                             min_samples_split = hiperparametry["min_samples_split"])\n    las.fit(X = X_train, y = y_train)\n    głupi = DummyClassifier(strategy="stratified", random_state=42)\n    głupi.fit(X = X_train, y = y_train)\n    rezultaty = rezultaty.append({"dokładność_las": las.score(X = X_test, y = y_test), \n                                  "dokładność_głupi": głupi.score(X = X_test, y = y_test)}, ignore_index = True)')


# In[31]:


rezultaty


# In[32]:


rezultaty["dokładność_las"].hist()


# In[33]:


rezultaty["dokładność_głupi"].hist()


# In[34]:


rezultaty.plot.hist(alpha=0.5)


# In[35]:


pandas_profiling.ProfileReport(rezultaty)


# In[36]:


#Las - czy większy znaczy lepszy?


# In[37]:


#zwiekszamy liczbę drzew do 100


# In[38]:


get_ipython().run_cell_magic('time', '', 'rezultaty = pd.DataFrame()\nfor ziarno_losowośći in range(0,100):\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=ziarno_losowośći, stratify = y)\n    las = RandomForestClassifier(criterion = "entropy", random_state = 42, bootstrap = True, max_features = max_features, \n                             max_depth = hiperparametry["max_depth"], \n                             min_samples_leaf = hiperparametry["min_samples_leaf"], \n                             min_samples_split = hiperparametry["min_samples_split"], n_estimators = 100)\n    las.fit(X = X_train, y = y_train)\n    głupi = DummyClassifier(strategy="stratified", random_state=42)\n    głupi.fit(X = X_train, y = y_train)\n    rezultaty = rezultaty.append({"dokładność_las": las.score(X = X_test, y = y_test), \n                                  "dokładność_głupi": głupi.score(X = X_test, y = y_test)}, ignore_index = True)')


# In[39]:


rezultaty["dokładność_las"].hist()


# In[40]:


pandas_profiling.ProfileReport(rezultaty)


# In[41]:


#Feature importances


# In[42]:


las.feature_importances_ # informacja o tym, jak dana cecha przyczyniła się do dokonania predykcji


# In[43]:


X_train.columns


# In[44]:


las.feature_importances_.sum()


# In[45]:


a=zip(X_train.columns,las.feature_importances_)


# In[46]:


set(a)


# In[47]:


# OOB score - out of bag score. Ponieważ używamy bootstrap to dane treningowe są losowane do drzew decyzyjnych. 
# potem do oceny algorytmu decyzyjnego brane są pod uwagę dane ze zbioru treningowego, których to drzewo nigdy nie widziało.
#bootstrap - drzewo nie widzi całego zbioru testowego, tylko losowe dane treningowe


# In[48]:



las = RandomForestClassifier(criterion = "entropy", random_state = 42, bootstrap = True, max_features = max_features, 
                             max_depth = hiperparametry["max_depth"], 
                             min_samples_leaf = hiperparametry["min_samples_leaf"], 
                             min_samples_split = hiperparametry["min_samples_split"], n_estimators = 100, 
                             oob_score = True)


# In[49]:


las.fit(X=X_train,y=y_train)


# In[50]:


las.oob_score_


# In[51]:


las.score(X=X_test,y=y_test)


# In[52]:


#Wizualizacja lasu losowego jako wizualizacja pojedynczych drzew


# In[53]:


from sklearn import tree
import graphviz


# In[54]:


for numerek in range(0,11):
    drzewo = las.estimators_[numerek]
    dot_data = tree.export_graphviz(drzewo, out_file = None,
                         feature_names = X.columns, class_names = ["Z", "Ł"],
                         filled = True, rounded = True,  
                         special_characters = True)  
    graph = graphviz.Source(dot_data)  
    graph.render("..\output\drzewo-{}".format(numerek))

