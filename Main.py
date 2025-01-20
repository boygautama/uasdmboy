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

     # Sex
     sex_counts = data['Sex'].value_counts().reset_index()
     sex_counts.columns = ['Sex', 'Count']
     fig_sex = px.bar(sex_counts, x='Sex', y='Count', labels={'Sex': 'Sex', 'Count': 'Count'}, title='Sex Distribution')
     st.plotly_chart(fig_sex)
    
    # BP
     bp_counts = data['BP'].value_counts().reset_index()
     bp_counts.columns = ['BP', 'Count']
     fig_bp = px.bar(bp_counts, x='BP', y='Count', labels={'BP': 'Blood Pressure', 'Count': 'Count'}, title='Blood Pressure Distribution')
     st.plotly_chart(fig_bp)
        
    # Cholesterol
     cholesterol_counts = data['Cholesterol'].value_counts().reset_index()
     cholesterol_counts.columns = ['Cholesterol', 'Count']
     fig_cholesterol = px.bar(cholesterol_counts, x='Cholesterol', y='Count', labels={'Cholesterol': 'Cholesterol', 'Count': 'Count'}, title='Cholesterol Distribution')
     st.plotly_chart(fig_cholesterol)


with st.expander('Modelling'):
    st.write('Splitting')
    data=data.drop(columns=['Sex','BP', 'Cholesterol','Drug'])
    st.write(f'Dataset : {data.shape}')
    
    X_train,X_test,y_train,y_test = train_test_split(data.drop(['Na_to_K'],axis=1), 
                                                     data['Na_to_K'],
                                                     test_size=0.30)
    
    st.success('Apply Random Forest Regressor')
    rf_regressor = RandomForestRegressor(max_depth=2,random_state=0)
    rf_regressor.fit(X_train,y_train)
    #make prediction
    y_pred_rf = rf_regressor.predict(X_test)
    score = mean_absolute_error(y_test, y_pred_rf)
    st.write(score)


     # Sidebar
    st.sidebar.header('Informasi Pasien')
    age = st.sidebar.slider('Umur', 15, 75, 50)
    sex = st.sidebar.selectbox('Kelamin', ['Female', 'Male'])
    bp = st.sidebar.selectbox('Tekanan Darah', ['LOW', 'NORMAL', 'HIGH'])
    cholesterol = st.sidebar.selectbox('Cholesterol', ['NORMAL', 'HIGH'])
    na_to_k = st.sidebar.slider('Na_to_K', 5.0, 40.0, 15.0)

     # Prediction
new_patient = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'BP': [bp],
    'Cholesterol': [cholesterol],
    'Na_to_K': [na_to_k]
})

st.write("Prediksi Data Pasien:")
st.write(new_patient)

required_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
if all(column in data.columns for column in required_columns):
    # Predict the customer group
    X = data[required_columns]
    predictions = rf_regressor.predict(new_patient).reshape(1,-1)
    data['Drug_pred'] = predictions

    st.write("Predictions:")
    st.write(data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug', 'Drug_pred']])
