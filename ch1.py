import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Housing.csv")

st.title("House Price Prediction")
st.header("Your one-stop solution to find the ideal house")
st.image('house.png')
# st.write(df.head(10))

cols=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
df[cols]=df[cols].replace({'yes':1,'no':0})
df['furnishingstatus']=df['furnishingstatus'].replace({'furnished':1,'unfurnished':0,'semi-furnished':0.5})

# remove outliers from price
# Outliers in the target hurt the model more
price_mean = df['price'].mean()
price_std = df['price'].std()
lower_bound = price_mean - 3 * price_std
upper_bound = price_mean + 3 * price_std
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# Feature Engineering
df['price_per_sqft'] = df['price'] / df['area']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']+ df['guestroom']


#scalling features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_cols = ['area', 'price_per_sqft', 'total_rooms']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])


#model test train split 
from sklearn.model_selection import train_test_split
X=df.drop('price',axis=1)
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#train model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)
# print(r2_score(y_test,y_pred))


st.subheader('Tell me your specifications')
price_per_ft=st.number_input('Enter the price_per_square fit')
total_rooms=st.number_input('enter the number of rooms')
area=st.number_input('enter the area')
bedrooms=st.number_input('enter the number of bedrooms you want')
bathrooms=st.number_input('enter the number of bathrooms you want')
stories=st.number_input('enter the number of stories you have envisioned for your house')
mainroad=st.radio('Do you want a view of the mainroad',['Yes','No'])
guestroom=st.radio('Do you want a guestroom',['Yes','No'])
basement=st.radio('Do you want a basement',['Yes','No'])
hotwaterheating=st.radio('Do you want hotwaterheating',['Yes','No'])
airconditioning=st.radio('Do you want airconditioning',['Yes','No'])
parking=st.number_input('How many vehicles do you have just to get an idea about the parking space')
prefarea=st.radio('do you want a prefarea',['yes','No'])
furnishing_status=st.selectbox("What kind of furnishing do you want?",['furnished','semi-furnished','unfurnished'])

price_per_ft = float(price_per_ft)
total_rooms = float(total_rooms)
area = float(area)
bedrooms = float(bedrooms)
bathrooms = float(bathrooms)
stories = float(stories)
parking = float(parking)

# replace is only used for pandas or dataframe not for a single string
# furnishing_status=furnishing_status.replace({'furnished':1,'unfurnished':0,'semi-furnished':0.5})
furnishing_status = {'furnished': 1, 'unfurnished': 0, 'semi-furnished': 0.5}[furnishing_status]

# cols=[mainroad,guestroom,basement,hotwaterheating,airconditioning,prefarea]
# cols=cols.replace({'yes':1,'no':0})

mainroad = 1 if mainroad == 'Yes' else 0
guestroom = 1 if guestroom == 'Yes' else 0
basement = 1 if basement == 'Yes' else 0
hotwaterheating = 1 if hotwaterheating == 'Yes' else 0
airconditioning = 1 if airconditioning == 'Yes' else 0
prefarea = 1 if prefarea == 'Yes' else 0


new_req=np.array([[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishing_status,price_per_ft,total_rooms]])


price=reg.predict(new_req)
st.write(f"according to the description given by you,the price of house is {price}")
         




