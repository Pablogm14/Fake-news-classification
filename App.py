
import streamlit as st 
import plotly.express as px 


#import packages
import numpy as np
import pandas as pd


import torch

from transformers import AutoTokenizer, AutoModel
import re

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

import os

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
import pickle as pkl
from sklearn import *

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC
from streamlit import caching
caching.clear_cache()

import joblib 

import requests
from bs4 import BeautifulSoup

import gc
#Enable garbage collection
gc.enable()
gc.collect()
pd.options.display.max_colwidth=800
pd.options.display.max_columns=None

path = os.path.dirname(__file__)
from textos import texto_titulo,texto_subtitulo

st.title(texto_titulo)
st.header(texto_subtitulo)
@st.cache(suppress_st_warning=True)
def scraping(url):
    page = requests.get(url)
#Con el mundo, marca, el pais funciona

    soup = BeautifulSoup(page.content, 'html.parser')
    sopa=soup.find_all('p')
    a=sopa[1].get_text()
    for i in np.arange(2,len(sopa)-1):
        a=a+sopa[i].get_text()
    if re.findall(r'eldiario.es', url)!=[]:
        a=a[397:]
        
    return a
#INTRODUCIR LA NOTICIA
side=st.sidebar
with side:
    user_input = st.text_area(label="Introduzca la noticia o una url de esta aquí",height=15, value="La masacre de Ayotzinapa, tres años sin respuestas. La desaparición de los *NUMBER* estudiantes en México cumple *NUMBER* meses sin que el Gobierno haya sido capaz de desenredar el caso. Este martes *NUMBER* de septiembre México ha amanecido con una acumulación de tragedias sin resolver. Una semana después de que un terremoto de magnitud *NUMBER* cimbrara el centro del país y se llevara por delante la vida de al menos *NUMBER* personas, bajo la corteza mexicana se encuentran ocultas todavía las respuestas sobre la desaparición de los *NUMBER* estudiantes de Ayotzinapa hace hoy tres años. Uno de los sucesos más graves de la historia del país, que abrió profundas cicatrices en la imagen del Gobierno mexicano y puso a la nación en el punto de mira internacional, cumple años sin una sola sentencia condenatoria por lo ocurrido y con las mismas dudas: ¿Qué pasó? ¿Dónde están?. Durante este tiempo la investigación se ha enredado tanto que la información al respecto se ha vuelto abrumadora. Hay tantas versiones, a veces contradictorias, que resulta laberíntico resolver cuándo ocurrió tal cosa, cómo ocurrió y dónde. O qué pasó justo después con cada uno de los implicados. Hasta la fecha se han detenido a más de *NUMBER* sospechosos, pero no se ha emitido ninguna sentencia condenatoria. Y solo cuatro están acusados de homicidio, tentativa de homicidio y ninguno por desaparición forzosa. Únicamente los restos de un estudiante pudieron ser identificados. Sobre los otros *NUMBER*, no ha sido posible determinar dónde están o qué les ocurrió.. La teoría oficial fue desde el primer momento que los estudiantes fueron asesinados por narcos de Iguala, una capital comarcal de Guerrero, en el México profundo, e incinerados en un basurero en medio del monte. No obstante, ante las dudas que planteaba esta hipótesis, un grupo de cinco especialistas extranjeros, designados por la Comisión Interamericana de Derechos Humanos, conocidos como el GIEI (Grupo Interdisciplinario de Expertos Independientes), analizó las pruebas e hizo sus propias pesquisas. Concluyó que esa versión no se sostenía con hechos y que dependía de confesiones de detenidos que pudieron haber testificado bajo tortura.. Durante un año los expertos internacionales se encargaron de recabar testimonios, revisar documentos y vídeos de lo ocurrido aquellos días de *NUMBER*. Su posición crítica respecto a temas polémicos, como el papel del Ejército —a quien se le acusó de no actuar durante la cacería contra los estudiantes— incomodaron al Ejecutivo. En su informe final presentado en abril del *NUMBER* el GIEI acusó dilaciones, obstrucciones y bloqueos del Gobierno mexicano al trabajo realizado.. El Gobierno y los expertos ni siquiera coincidieron en la causa de la masacre. Según la versión oficial, la noche del *NUMBER* al *NUMBER* de septiembre, la policía local de Iguala se lanzó a una feroz persecución de los autobuses de los estudiantes. Tras la cacería aparecieron seis cadáveres y desaparecieron *NUMBER* alumnos. La explicación del Gobierno fue que el alcalde, mafioso en jefe del municipio, dio una orden de escarmiento a aquellos jóvenes marxistas que derivó en una escabechina: la policía los detiene, los entrega a los narcos y estos, confundiéndolos con narcos rivales, optan por el exterminio. Los matan. Los queman en una pira de neumáticos y madera. Tiran sus cenizas a un río.. El GIEI negó que los cuerpos de los estudiantes hubieran sido quemados en el basurero y resaltó que el batallón militar de la zona vio la persecución y detención de los estudiantes. No creyó en la teoría de la orden del alcalde y planteó la sospecha de que uno de los autobuses —líneas de pasajeros tomadas a la fuerza por los estudiantes para ir a una manifestación en Ciudad de México— llevara en sus tripas un alijo de heroína sin que ellos lo supieran, que los señores de la droga no quisieron perder y cuyo desvío castigaron con ira. Los expertos internacionales pidieron entrevistar a los soldados del batallón y nunca se lo concedieron. ""Dentro del aparato del Estado hay fuerzas que no quieren que se investigue la verdad. Son fuerzas estructurales"", afirmaba en una entrevista a este diario el español Carlos Beristáin, integrante del grupo.. La Procuraduría General de la República ha detallado este martes en un comunicado que ""en todo momento se ha obrado con objetividad en el caso, cuya investigación constituye la más amplia realizada en época alguna por la PGR. Los más de *NUMBER* tomos del expediente así lo confirman"". Alega que hay más de *NUMBER* detenidos en la cárcel, *NUMBER* procesados por secuestro y que si no están sentenciados es porque están agotando todas las vías legales para retrasar el juicio. A tres años de distancia de los hechos de Iguala el Gobierno de la República reafirma su compromiso con las víctimas y reitera que continuará agotando todos los medios a su alcance"", puntualiza el organismo. *NUMBER* meses después de lo sucedido, después de un agotamiento gradual de las protestas civiles, las cuestiones principales de la tragedia se encuentran estancadas en el punto de partida: ¿Dónde están? ¿Qué pasó?")
    if user_input[0:4]=="http":
        user_input=scraping(user_input)

#ELEGIR MODELO DE PREPROCESADO
with side:
    option = st.selectbox(
     'Elija el modelo de preprocesado de datos',
     (' ','Beto', 'Beto emotion')) #'Beto sentiment', 'Multilingual bert', 'Distilbert', 'spanberta'))

#ELEGIR EL MODELO DE CLASIFICACIÓN
with side:
    option2 = st.selectbox(
     'Elija el algoritmo de clasificación deseado',
     (' ','Regresión Logística',  'Random Forest', 'Redes Neuronales'))# 'K-NN', 'Árbol de decisión', 'Bagging','XGBoost', 'SVM'))

with side:
    boton=st.button('Ejecutar')


    #Creamos las funciones pertinentes para cargar los datos
@st.cache(suppress_st_warning=True)
def clean_fake(text):
    cleaned_text_1 = re.sub('".*?"', '', text)
    cleaned_text = re.sub(r'\.(?=[^ \W\d])', '. ',  cleaned_text_1)
    return cleaned_text
@st.cache(suppress_st_warning=True)
def clean(text):
# removing all the characters other than alphabets
    cleaned_text_1= re.sub("[^a-zA-ZñÑ]", " ", text)
    cleaned_text_2 = re.sub(r'\W+', ' ', cleaned_text_1)
        # converting text to lower case
    cleaned_text = re.sub("\d+", " ", cleaned_text_2)
        #all lowercase
    cleaned_text= cleaned_text.lower()
    return cleaned_text
@st.cache(suppress_st_warning=True)
def normalize(s):  #Para quitar las tildes
    replacements = (
            ("á", "a"),
            ("é", "e"),
            ("í", "i"),
                ("ó", "o"),
                ("ú", "u"),
                )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s
        
        
noticia=normalize(user_input)
noticia= clean_fake(noticia)
        
        
    #clean text
noticia=clean(noticia)
        


@st.cache(suppress_st_warning=True)
def stop_words_fun(col):
        text = col 
        string = str(text)
        text_tokens = word_tokenize(string) #Separa las palabras y los símbolos
        tokens_without_sw = [word for word in text_tokens if not word in stop]
        joined = " ".join(tokens_without_sw)
        return joined
    

    #remove stopwords
stop = stopwords.words('spanish') 
stop = set(stop)
@st.cache(suppress_st_warning=True)
def remove_stop_words(col):
    text = str(col)
    sent_text = nltk.sent_tokenize(text)
    return sent_text
    


noticia =  remove_stop_words(noticia)
    
noticia = stop_words_fun(noticia) #Elimino preposiciones, etc

noticia=clean(noticia)

if option==' ' or option2==' ':
    st.write("Selecciona un modelo")
else:
    @st.cache(suppress_st_warning=True)
    def resultados(noticia):
        @st.cache(suppress_st_warning=True)
        def modelo_bert(nombre):
            tokenizer = AutoTokenizer.from_pretrained(nombre)
            model = AutoModel.from_pretrained(nombre)
            tokenized =tokenizer.encode(noticia, add_special_tokens=True, truncation = True, max_length = 512)
            input_ids = torch.tensor(tokenized)  
    
            input_ids = input_ids.unsqueeze(0)

            with torch.no_grad():
                last_hidden_states = model(input_ids)
    
            return last_hidden_states
    
        if option=='Beto':
            token=modelo_bert("dccuchile/bert-base-spanish-wwm-uncased")
            features = token[0][:,0,:].numpy()
        if option=='Beto emotion':
            token=modelo_bert("finiteautomata/beto-emotion-analysis")
            features = token[0][:,0,:].numpy()
        if option=='Beto sentiment':
            token=modelo_bert("finiteautomata/beto-sentiment-analysis")
            features = token[0][:,0,:].numpy()
        if option== 'Multilingual bert':
            token=modelo_bert("nlptown/bert-base-multilingual-uncased-sentiment")
            features = token[0][:,0,:].numpy()
        if option=='Distilbert':
            token=modelo_bert("mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")
            features = token[0][:,0,:].numpy()
        if option=='spanberta':
            token=modelo_bert("skimai/spanberta-base-cased")
            features = token[0][:,0,:].numpy()
       #if option=='TFIDF' or option=='conteo':
     
           # if option=='TFIDF':
                #vectorizer = joblib.load(path+"/TFIDF/TFIDF Temas/TFIDFTEMAS.pkl")
               # features = vectorizer.transform(pd.Series(noticia)).toarray()
            #if option=='conteo':
               # vectorizer = joblib.load(".\conteo\CONTEOTEMAS.pkl")
               # features = vectorizer.transform(pd.Series(noticia)).toarray()
    
        
        
        
    
    #vectorizer = TfidfVectorizer()
    #features = vectorizer.fit_transform([noticia]).toarray()
    #Cargo el modelo. 
    #Hago las predicciones en base a ese texto
        pipe_lr = joblib.load(open(path+"/"+option+"/"+ option+" Temas/"+option2.replace(" ", "")+option.replace(" ", "")+".pkl","rb"))
        if option2=="Bagging":
            pipe_lr.n_features_=pipe_lr.n_features_in_
   
   
        results = pipe_lr.predict(features)
        probs= pipe_lr.predict_proba(features)
        labels=pipe_lr.classes_
        return [results[0],probs[0],labels, features]


    import plotly.express as px
#Gráfica
    @st.cache(suppress_st_warning=True)
    def grafica(datos,etiquetas):

        df=pd.DataFrame()
        df["Probabilidad"]=datos
        df["Categoría"]=etiquetas
        fig = px.pie(df, values='Probabilidad', names='Categoría')
        fig.update_layout(title_text="Probabilidad por categoría",title_xref="paper",title_xanchor="center",title_font_family="Times New Roman",
                          legend_title_text="Categoría", title_x=0.53)
   
   
        return fig
    import eli5
    from eli5 import explain_weights, explain_prediction
    from eli5.lime import TextExplainer
    from eli5.lime.samplers import MaskingTextSampler
    from IPython.core.display import display, HTML
    from eli5.formatters import format_as_html, format_as_text, format_html_styles, fields
    show_html = lambda html: display(HTML(html))
    show_html_expl = lambda expl, **kwargs: show_html(format_as_html(expl, include_styles=False, **kwargs))
    show_html(format_html_styles())

    @st.cache(suppress_st_warning=True)
    def predictor(texts):
        return np.array([resultados(string)[1] for string in texts])

    te = TextExplainer(n_samples=5,random_state=42,sampler=MaskingTextSampler())
    te.fit(noticia, predictor)

    RES=resultados(noticia)

    expl = te.show_prediction(target_names=[RES[2][0], RES[2][1], RES[2][2], RES[2][3],
                                            RES[2][4]], top_targets=5, show_feature_values=True)
    raw_html = expl._repr_html_()


    if boton==True:    
        with st.spinner('Calculando...'):
            st.write(grafica(RES[1],RES[2]))
            st.write("La temática de la noticia es:", RES[0], "con una probabilidad igual a", format( RES[1].max(), '.2%'))

    
        from streamlit.components import v1
        with st.expander('IMPORTANCIA DEL TEXTO EN CADA CATEGORÍA'):
            v1.html(raw_html, height=10000)
    
    
        #CLASIFICACIÓN FAKE NEWS
        Categoria=RES[0]
        FEATURES=RES[3]
        path = os.path.dirname(__file__)
        @st.cache(suppress_st_warning=True)
        def resultados2(features,option,option2,Categoria):
            #if option=='TFIDF' or option=='conteo':
            #if option=='TFIDF':
               # vectorizer = joblib.load(path+"/TFIDF/TFIDF "+Categoria+"/TFIDF"+Categoria.replace(" ", "")+".pkl")
               # features = vectorizer.transform(pd.Series(noticia)).toarray()
           # if option=='conteo':
              #  vectorizer = joblib.load(path+"/conteo/conteo "+Categoria+"/conteo"+Categoria.replace(" ", "")+".pkl")
               # features = vectorizer.transform(pd.Series(noticia)).toarray()
            #else: features=features
            pipe_lr = joblib.load(open(path+"/"+option+"/"+ option+" "+Categoria+"/"+option2.replace(" ", "")+option.replace(" ", "")+Categoria.replace(" ", "")+".pkl","rb"))
            if option2=="Bagging":
                pipe_lr.n_features_=pipe_lr.n_features_in_
        
            results = pipe_lr.predict(features)
            probs= pipe_lr.predict_proba(features)
            labels=pipe_lr.classes_
        
            return [results[0],probs[0],labels]

        RES2=resultados2(FEATURES,option,option2,Categoria)
        @st.cache(suppress_st_warning=True)
        def grafica2(datos,etiquetas):
            df=pd.DataFrame()
            df["Probabilidad"]=datos
            df["Categoría"]=["Falsa", "Verdadera"]
            fig = px.bar(df, x="Categoría", y="Probabilidad")
            fig.update_layout(title_text="Probabilidad por categoría",title_xref="paper",title_xanchor="center",title_font_family="Times New Roman",
                          legend_title_text="Categoría", title_x=0.53)
            return fig

        a=[]
        if RES2[0]==True:
            a='Verdadera'
        else: a='Falsa'
        with st.expander('VERACIDAD DE LA NOTICIA ATENDIENDO A LA TEMÁTICA OBTENIDA'):
            with st.spinner('Calculando...'):
                st.write(grafica2(RES2[1],RES2[2]))   
                st.write("La noticia es:", a , "con una probabilidad igual a", format( RES2[1].max(), '.2%'))
    else:
        st.write("Introduzca los parámetros y ejecute la aplicación")
        
        
import gc
gc.collect()