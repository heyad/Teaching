import streamlit as st 
from PIL import Image
import numpy as np
import pickle
import cv2 # computer vision library 
import pandas as pd
from random import randint
import plotly.express as px
import plotly.graph_objects as go


# define some strings to for the title and intro page of the app
title = "<h1 style='text-align: left; color: Tomato;'>"
title+= "Symbols in Engineering Drawings<br></h1>"
title+= "<h2 style='text-align: left; color: Gray;'>"
title+= "<b>Random Forest</b></h2><br>"

intro_msg = '<p align="justify"><b>Random Forest</b> Classifier is used to build a model'
intro_msg+= ' that can learn the class of symbols of engineering diagrams'
intro_msg+= '.  These symbols appear in engineering diagrams such as Piping'
intro_msg+= 'and Instrumentation Diagrams (<b>P & ID</b>), and they'
intro_msg+= 'are very common in the Oil and Gas industry.<br><br>'

st.markdown(title,True)
st.markdown(intro_msg,True)

# a function to load the Random Forest Model
@st.cache(allow_output_mutation=True)
def get_model():
    pickle_file = open('rf_model.pkl', 'rb') 
    model = pickle.load(pickle_file)
    return model 

# # read the testing set from a data frame
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('../data/Symbols.csv')
    # split features and labels into two differen dataframes (X, y)
    features = df.loc[:, df.columns != 'label']
    labels = df['label'].to_numpy()

    return df,features,labels


# the symbols size (as in our model - see lab notes)
image_w =100
image_h =100

# store data, features and labels 
df,features,labels = load_data()

# Below is the code to use the model to make predictions, notice two things:
# 1. I am doing predictions on the whole dataset (not a good idea), because 
# we use part of it for training the model
# 2. I do it in one line of code alternatively, you can first:
# model = get_model(), and then 
# model.predict(...)

# Notice also that I have converted the features dataframe into numpy array

preds = get_model().predict(np.array(features))



# simple function to combine features, labels and predictions in one data frame
def features_labels_preds():
    results = pd.DataFrame({'Actual':df['label'],'Predicted':preds})
    df_results = pd.concat([features, results], axis=1)

    # lets compare predictions against actual class labels
    check_predictions = np.where(df_results['Actual'] == df_results['Predicted'], True, False)
    # lets add one more column to the above dataframe 
    df_results['isCorrect']=check_predictions

    return df_results

# Function to view numbers of rows/ from correct/incorrectly
# classified instances with the last few columns
def prepare_results(correct=True):

    show_results = df_results

    if correct==True:
        # return correctly classified instances 
        rows_to_show = show_results[show_results.isCorrect==True]


    else:
        # return incorrectly classified instances 
        rows_to_show = show_results[show_results.isCorrect!=True]
    return rows_to_show

# Geneate an image from the set of features in the dataset
def gen_image(arr):
    two_d = (np.reshape(arr, (image_w, image_h)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')

    # resize for viewing purpose
    n_w = image_w * 2
    n_h = image_h * 2
    img = img.resize((n_w,n_h),Image.ANTIALIAS)

    return img

# function to show distribution of incorrectly classified examples
# it returns plotly figure
def visualise_correct_inst(Correct_classified=True):

    if Correct_classified == True:
        correct_insts = pd.DataFrame(df_results[df_results.isCorrect==True].groupby(['Actual'])['isCorrect'].count()).reset_index()
        correct_insts.columns = ['Symbol','Count']
        fig = go.Figure(data=[go.Bar(
            x=correct_insts['Symbol'],
            y=correct_insts['Count'],
            text=correct_insts['Count'],
            )])
        number_of_correctly_classified = correct_insts['Count'].sum()
        msg=str(number_of_correctly_classified)+' instances were correctly classified'
        fig.update_layout(template='plotly_white')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        fig.update_yaxes(title_text="Count", hoverformat=".3f")
        fig.update_layout(title_text=msg, title_x=0.5)
        return fig
    else:
        correct_insts = pd.DataFrame(df_results[df_results.isCorrect!=True].groupby(['Actual'])['isCorrect'].count()).reset_index()
        correct_insts.columns = ['Symbol','Count']
        fig = go.Figure(data=[go.Bar(
            x=correct_insts['Symbol'],
            y=correct_insts['Count'],
            text=correct_insts['Count'],
            marker_color='red',
            )])
        number_of_correctly_classified = correct_insts['Count'].sum()
        msg=str(number_of_correctly_classified)+' instances were incorrectly classified'
        fig.update_layout(template='plotly_white')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
        fig.update_yaxes(title_text="Count", hoverformat=".3f")
        fig.update_layout(title_text=msg, title_x=0.5)
        return fig


# function to show the image
def view_image(image_index = 0):
    # notice we convert it to numpy array and reshape it

    # extract the row in the dataset at index image_index
    image_to_show = features.iloc[image_index].to_numpy()
    # this step is needed for viewing the imgage to get values 
    #between 0 and 1 representing colors
    image_to_show = image_to_show/255.0
    image_to_show = gen_image(image_to_show)
    # Get the label of the corresponding image
    label= labels[image_index]
    # return image to be viewed in the app and the label
    return image_to_show, label

# show class distribution
show_class_dist = st.checkbox('Show Class Distribution')
if show_class_dist:
    fig = px.histogram(df, x="label",width=800,height=500)
    #fig.update_layout(xlabel='x')
    st.plotly_chart(fig)



show_symbol = st.checkbox('Show Symbols')
if show_symbol:
    # prepare message to appear on the web app page
    msg = 'The dataset contains more than <b>'+str(len(df['label'].unique()))
    msg+='</b> different symbols of <b>P&ID</b> Diagrams.'
    msg+=' You can explore the shapes of these symbols by clicking the sidebar'

    st.markdown(msg,True)
    #st.write('Check the Slidebar to navigate through images in your test set')
    symbol_type = st.sidebar.selectbox('Select Type',df['label'].unique())
    
    symbol_name = "<h1 style='text-align: left; color: Gray;'>"
    symbol_name+=str(symbol_type+'<br><br>')
    st.markdown(symbol_name,True)
    # get the index of the first row of the subset of similar symbols 
    image_index = df.loc[df.label==symbol_type][:1].index
    # prepare the image for viewing (return image, and label)


    image_v, label = view_image(image_index)
    st.image(image_v, caption='', use_column_width=False)


# correctly classified symbols

n = 10
df_results = features_labels_preds()

clf_results = st.checkbox('Show Classification Results')
if clf_results: 
    
    right_wrong_clf = st.sidebar.radio("Show Classified Instances", 
    ['Correctly Classified','Incorrectly Classified'])
    # add sidebar field to input number of records by the user 
    n = st.sidebar.number_input('How many predictons to view',0,df.shape[0],5)
    # show correctly or incorrectly instances based on user's choice
    if right_wrong_clf =='Incorrectly Classified':
        st.plotly_chart(visualise_correct_inst(False))
        # show how instances were misclassified (n) is entered by the user
        rows_to_show = prepare_results(False)
        msg = "<h4 style='text-align: center; color: red;'>"
        msg+=str(rows_to_show.shape[0])+' were incorrectly classified'
        st.markdown(msg,True)
        st.write(rows_to_show.iloc[:n,-5:])

    else: 
        st.plotly_chart(visualise_correct_inst())
        rows_to_show = prepare_results(True)
        msg = "<h4 style='text-align: center; color: blue;'>"
        msg+=str(rows_to_show.shape[0])+' were correctly classified'
        st.markdown(msg,True)
        st.write(rows_to_show.iloc[:n,-5:])

    # prepare message to appear on the app page
    msg= '<b> Notice</b> that Random Forest is used here to classify the whole dataset.'
    msg+= ' It has <b>'+str(df_results.shape[0]) + ' </b>records'
    msg+='. This is not a good idea, because some of the data was used in the training.'
    msg+=' However, we just do it here for illustration purposes.'
    msg+=' Details about train and testing can be found in the lab document.'
    
    st.markdown(msg,True)


# barplot 















