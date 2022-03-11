#!/usr/bin/env python
# coding: utf-8

# In[34]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st


# In[35]:


# Titel en subtitel van de app

st.title('Diabetes detectie')
st.header('Kijk of u diabetes heeft met behulp van deze app')

# Teskst behorende bij de app

st.write('Diabetes is een chronische stofwisselingsziekte. Bij diabetes zit er te veel suiker in het bloed. Het lichaam kan de bloedsuiker niet op peil houden. Met behulp van deze machine learning web app wordt het mogelijk om aan de hand van ingevoerde parameters een diagnose te maken over de mogelijkheid dat iemand diabetes heeft.De app maakt gebruik van historische data om de kans op diabetes te calculeren. Dit kan mensen helpen om betere en snellere diagnoses te maken of mensen helpen die geen tijd of geld hebben om een doctor te bezoeken.')

#Voeg afbeelding toe > werkt alleen als de persoon die het upload naar streamlit zelf de afbeelding opslaat en filepath noteerd naar de afbeelding
#Image = Image.open("C:\Users\joshua.bierenbrood\Documents\Data Science\Intro to datascience\Werkcollege week 3\diabetes.jpg")
#st.image(image, caption = 'ML', use_column_width = True)


# In[36]:


df = pd.read_csv('diabetes.csv')


# In[37]:


st.title("Diabetes Database")


# In[38]:


df.head()


# In[39]:


df.info()


# In[40]:


df.eq(0).sum()
#veel 0 waarden, vervangen door NaN en dan opvullen met gemiddelden


# In[41]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI',
    'DiabetesPedigreeFunction','Age']]= df[['Glucose','BloodPressure',
    'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)


# In[42]:


df.fillna(df.mean(), inplace=True)


# In[43]:


df.eq(0).sum() #checken


# In[44]:


df.describe()


# In[45]:


st.subheader('Informatie over de data:')
st.write('Voor het maken van deze app is een dataset over diabetes van kaggle gebruikt. De dataset is oorspronkelijk van het National Institute of Diabetes and Digestive and Kidney Disease. Het is verkregen uit een onderzoek onder 768 vrouwen boven de 21 jaar.')
st.dataframe(df)


# In[46]:


st.title("Correlatie tussen de verschillende variabelen")
st.write('In de onderstaande heatmap is te zien hoe de variabelen van elkaar afhangen.')
fig, ax = plt.subplots()
sns.heatmap(df.corr(), ax=ax, annot=True)
st.write(fig)


# In[47]:


glucose = df['Glucose'].hist()


# In[48]:


zwangerschappen = df['Pregnancies'].hist()


# In[49]:


bloeddruk = df['BloodPressure'].hist()


# In[50]:


huiddikte = df['SkinThickness'].hist()


# In[51]:


insuline = df['Insulin'].hist()


# In[52]:


BMI = df['BMI'].hist()


# In[53]:


DPF = df['DiabetesPedigreeFunction'].hist()


# In[54]:


leeftijd = df['Age'].hist()


# In[55]:


outcome = df['Outcome'].hist()


# In[56]:


sns.distplot(df['Glucose'])


# In[57]:


st.title('Visualisatie van de data')
st.write('Hieronder kunt u de verdeling per variabele in de dataset zien.')
option = st.selectbox(
     'Welke variabele wilt u visualiseren?',
     ('glucose', 'zwangerschappen', 'bloeddruk', 'huiddikte', 'insuline', 'BMI', 'DPF', 'leeftijd'))
if option == 'glucose':
    st.bar_chart(df['Glucose'])
elif option == 'zwangerschappen':
    st.bar_chart(df['Pregnancies'])
elif option == 'bloeddruk':
    st.bar_chart(df['BloodPressure'])
elif option == 'huiddikte':
    st.bar_chart(df['SkinThickness'])
elif option == 'insuline':
    st.bar_chart(df['Insulin'])
elif option == 'BMI':
    st.bar_chart(df['BMI'])
elif option == 'DPF':
    st.bar_chart(df['DiabetesPedigreeFunction'])
elif option == 'leeftijd':
    st.bar_chart(df['Age']) 


# In[58]:


#Machine learning
x_data= df.drop('Outcome',axis=1)
y_data= df['Outcome']


# In[59]:


x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.3, random_state=42)


# In[60]:


def get_input():
    pregnancies = st.sidebar.slider('Zwangerschappen', 0, 17,0)
    glucose = st.sidebar.slider('Glucose', 44, 199, 44)
    blood_pressure = st.sidebar.slider('Bloeddruk',24, 122,24)
    skin_thickness = st.sidebar.slider('Huiddikte', 7, 99, 7)
    insulin = st.sidebar.slider('Insuline', 14.0, 846.0, 14.0)
    BMI = st.sidebar.slider('BMI', 18.2, 67.1, 18.2)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.078)
    age = st.sidebar.slider('Leeftijd', 21, 81, 21)
    
    user_data = {'Zwangerschappen': pregnancies,
                 'Glucose': glucose,
                 'Bloeddruk': blood_pressure,
                 'Huiddikte': skin_thickness,
                 'Insuline': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'Leeftijd': age
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features

gekozen_input = get_input()

st.subheader('Doe zelf de test of u diabetes heeft!')
st.write('Selecteer de voor u geldende parameters in de sliderboxen aan de linkerkant om de app te laten bepalen of u diabetes heeft.')
 
st.write('Om te kunnen classificeren of u wel of geen diabetes heeft maakt deze app gebruik van een Random Forest algoritme. Random forest wordt gezien als een hoog accurate en robuuste methode voor een classificatie probleem gezien het aantal decision trees dat deelneemt in het proces.')   
    
st.subheader('Uw gekozen waarden:')
st.write(gekozen_input)


# In[61]:


RandomForest=RandomForestClassifier()
RandomForest.fit(x_train, y_train)
Predict= RandomForest.predict(x_test)
print(Predict)


# In[62]:


accuracyRFC = accuracy_score(y_test, Predict)
print("Accuracy with Random Forrest Classification:", accuracyRFC)


# In[63]:


st.subheader('Accuratiescore bij het model')
st.write(str(accuracy_score(y_test, Predict) * 100) + '%')


# In[64]:


diabetes_ja_nee = RandomForest.predict(gekozen_input)


# In[65]:


st.subheader('Wel of geen diabetes?')
st.write("Wanneer de uitslag 1 is, heeft u wel diabetes. Wanneer de uitslag 0 is, heeft u geen diabetes.")
st.write(diabetes_ja_nee)


# In[66]:


st.subheader('Heeft u volgens de test diabetes?')
ja = st.checkbox('Ja')
nee = st.checkbox('Nee')

if ja:
     st.write('Wat vervelend voor u..')
if nee:
    st.write('Wat fijn voor u!')

