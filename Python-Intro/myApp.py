
import streamlit as st
import numpy as np
import pandas as pd

# set title
st.title('Adevrtising-Sales')

# paragraph to explain what is this app about 
st.markdown('This is first streamlit app. It will be used to explore the Advertising dataset and perform predictions of sales based on simple linear model. It must be noted that <b>linear</b> regression models are not best choice for this problem',True)
st.info('Use the sidebar controls to change the spending values on TV, Radio, and Newspaper Advertising, and also to control how many rows of the dataframe you want to show.')


#Simple function to read the data in the file. Notice that we cached this part 
@st.cache
def get_data():
    df = pd.read_csv('data/Adevrtising.csv')
    return df

# get the data into df data frame (pandas)
df = get_data()


# simple function to view the top n rows of the data frame
def show_df_rows(n = 10):
	st.write(df[:n])

# a function to do some basic Data Exploratory Analysis
def explore_df():
	n_rows = st.sidebar.number_input('Rows to View',1,df.shape[0],5)
	st.markdown('List of <b>' +str(n_rows)+'</b> Records'+' from'+ ' Data Frame <b>df</b>',True)
	show_df_rows(n_rows)
	st.markdown('The Data Frame has <b>' + str(df.shape[0]) + '</b> Rows, and <b>'+str(df.shape[1])+'</b> Columns',True)
	st.markdown('<b>'+str(n_rows) + '</b> Rows from the Data Frame are visible <b>',True)

# function to make predictions based on our LM model 

def predict_sales(tv = 10, radio = 10, newspaper = 10):

	# The numbers below from the lab document based on our lm model
	intercept = 2.938889369459412
	tv_coeff = 0.04576465  
	radio_coeff = 0.18853002 
	news_coeff = 0.00103749 # negative 
	# make prediction
	new_sales = intercept+ (tv_coeff*tv) +  (radio_coeff*radio)  -(news_coeff*newspaper)
	return new_sales

# lets add a checkbox to show or hide some records from our dataset
show_df = st.checkbox('Explore the Dataset',False)
if show_df:
	explore_df()

# controls for setting the value of spendings on tv, radio, ... 
# These will appear on the sidebar
tv = st.sidebar.slider('TV Spending',0,500)
radio = st.sidebar.slider('Radio Spending',0,500)
news = st.sidebar.slider('Newspaper Spending',0,500)

# Check box to perform prediction 
make_preds = st.checkbox('Make Prediction',False)
# check if make_preds is checked, then perform predictions
if make_preds: 

	sales = predict_sales(tv,radio,news)
	# output message
	st.markdown('Spending '+str(tv) +' units on <b>TV</b>, '+str(radio)+ ' units on <b>Radio</b>, '+ 'and '+str(news) + ' on <b>Newspaper</b> Advertising' +', will generate increase in sales by '+str(round(sales,3)) +' units',True)

