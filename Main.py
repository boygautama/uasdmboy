import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


st.title('Klasifikasi Data Kesehatan')
with st.expander('Informasi Dataset'):
    data = pd.read_csv('Classification.csv')
    st.write(data)

    st.success('Informasi Dataset')
    st.write(data.describe())
    
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

     # Bar plot of Sex
     sex_counts = data['Sex'].value_counts().reset_index()
     sex_counts.columns = ['Sex', 'Count']
     fig_sex = px.bar(sex_counts, x='Sex', y='Count', labels={'Sex': 'Sex', 'Count': 'Count'}, title='Sex Distribution')
     st.plotly_chart(fig_sex)
    
    # Bar plot of BP
     bp_counts = data['BP'].value_counts().reset_index()
     bp_counts.columns = ['BP', 'Count']
     fig_bp = px.bar(bp_counts, x='BP', y='Count', labels={'BP': 'Blood Pressure', 'Count': 'Count'}, title='Blood Pressure Distribution')
     st.plotly_chart(fig_bp)
        
    # Bar plot of Cholesterol
     cholesterol_counts = data['Cholesterol'].value_counts().reset_index()
     cholesterol_counts.columns = ['Cholesterol', 'Count']
     fig_cholesterol = px.bar(cholesterol_counts, x='Cholesterol', y='Count', labels={'Cholesterol': 'Cholesterol', 'Count': 'Count'}, title='Cholesterol Distribution')
     st.plotly_chart(fig_cholesterol)