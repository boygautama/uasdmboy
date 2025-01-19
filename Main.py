import streamlit as st
import pandas as pd 
import io
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


st.title('Klasifikasi Data Kesehatan')
with st.expander('Informasi Dataset'):
    data = pd.read_csv('Classification.csv')
    st.write(data)

    st.success('Informasi Dataset')
    data1 = pd.DataFrame(data)
    buffer = io.StringIO()
    data1.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.success('Analisa Univariat')
    deskriptif = data.describe()
    st.write(deskriptif)

   

with st.expander('VISUALISASI'):

     usia_min = data['Age'].min()
     usia_max = data['Age'].max()
     st.info('Rentang Usia : '+str(usia_min)+' tahun - '+str(usia_max)+' tahun')

     st.info('Visualisasi Per Column')

     fig,ax = plt.subplots()
     sns.histplot(data['Age'],color='blue', kde=True, fill=True)
     plt.xlabel('Usia')
     plt.ylabel('Jumlah')
     st.pyplot(fig)

     fig,ax = plt.subplots()
     sns.histplot(data['Drug'],color='red')
     plt.xlabel('Jenis Obat')
     plt.ylabel('Pengguna')
     plt.xticks(rotation=90)
     st.pyplot(fig)

     st.info('Korelasi Heatmap')
     kategori = ['Age','Na_to_K']
     matriks_korelasi = data[kategori].corr()
    
     fig,ax = plt.subplots()
     sns.heatmap(matriks_korelasi,annot=True, cmap='RdBu')
     plt.xticks(fontsize=8)
     plt.yticks(fontsize=8)
     plt.title('Korelasi Umur Terhadap N-K',fontsize=10)
     st.pyplot(fig)