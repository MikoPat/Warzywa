import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus



def decision(plant,humidity,growth,dryness):

  # czytanie pliku csv 
  df = pd.read_csv("data.csv")
  #print(df)
  
  z = {'CABBAGE': 2, 'PUMPKIN':4, 'MUSHROOM':3, 'CAULIFLOWER': 1}
  df['PLANT'] = df['PLANT'].map(z)
  
  #v = {'NO': 0, 'YES': 1}
  #df['VERMINS'] = df['VERMINS'].map(v)

  d = {'NO': 0, 'YES': 1}
  df['DEC'] = df['DEC'].map(d)

  #print(df)
  
  features_rest = ['PLANT','HUMIDITY','GROWTH','DRYNESS'] #dane, na których opiera się decyzja
  features_dec = ['DEC'] #kolumna z decyją

  X = df[features_rest]
  y = df[features_dec]

  #wyświetlkanie kolumn
  #print(X)
  #print(y)

  #tworzenie drzewa
  dtree = DecisionTreeClassifier()
  #przypisanie danych
  dtree = dtree.fit(X, y)
  #eksport drzewa
  r = export_text(dtree, feature_names=features_rest)
  
  print("\nDecision tree\n")
  print(r)

  a = dtree.predict([[plant,humidity,growth,dryness]])

  print("\n[1] means FEED THE PLANT")
  print("[0] means NOT FEED THE PLANT\n")

  print ("Decision for: ",plant,", feed: ", humidity,", growth: ", growth,", dryness:", dryness," is ", a,"\n")
  
    
plant = 1
humidity = 35
growth = 20
dryness = 12
decision(plant,humidity,growth,dryness)