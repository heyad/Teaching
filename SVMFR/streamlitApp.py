import streamlit as st 
from PIL import Image
import numpy as np
import pickle
import cv2 # computer vision library 
import pandas as pd

# a function to load the model (SVM)
@st.cache(allow_output_mutation=True)
def get_model():
    pickle_file = open('svm_model.pkl', 'rb') 
    model = pickle.load(pickle_file)
    return model 

# # read the testing set from a data frame
@st.cache(allow_output_mutation=True)
def read_test_set():
    df = pd.read_csv('Xy_test.csv')
    # split features and labels into two differen dataframes (X, y)
    features = df.loc[:, df.columns != 'Label']
    labels = df['Label'].to_numpy()

    return df,features,labels

# the target size (as in our model - see lab notes)
image_w =62
image_h =47

# list of names in the dataset 
faces = ['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush',
       'Gerhard Schroeder', 'Hugo Chavez', 'Junichiro Koizumi',
       'Tony Blair']

# load model, 
model = get_model()
# load test data, features (images), and labels 
df, features_df, labels = read_test_set()

# perform predictions using our pickled model
predictoins = model.predict(features_df)



# This function to prepare the image if uploaded from 
# a file to ensure it can be fed into our model

def prepare_image(uploaded_file):
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False)
    st.write('Original Image Size is :' + str(image.size))
    # convert image to grayscale 
    gs_image = image.convert(mode='L')

    # resize the image / similar to our trainign data
    gs_image = gs_image.resize((image_w,image_h))
    # flatten image into one row of pixels
    features=np.array(gs_image).flatten()
    # reshape the array to fit our model
    features = features.reshape(1,features.shape[0])
    return features

# Geneate an image from the set of features in the dataset
def gen_image(arr):
    two_d = (np.reshape(arr, (image_w, image_h)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')

    # resize for viewing purpose
    n_w = image_w * 2
    n_h = image_h * 2
    img = img.resize((n_w,n_h),Image.ANTIALIAS)

    return img

# function to show the image
def view_image(image_index = 0):
    # notice we convert it to numpy array and reshape it

    # extract the row in the dataset at index image_index
    image_to_show = features_df.iloc[image_index].to_numpy()
    # this step is needed for viewing the imgage to get values 
    #between 0 and 1 representing colors
    image_to_show = image_to_show/255.0
    image_to_show = gen_image(image_to_show)
    # Get the label of the corresponding image
    label= faces[labels[image_index]]
    # return image to be viewed in the app and the label
    return image_to_show, label

##############################################################################
################ This is the code to create the Streamlit App ################
##############################################################################

# set the title for your app
#st.title("Face Recgonition using SVM")
st.markdown("<h1 style='text-align: left; color: red;'>Face Recgonition using SVM</h1>", unsafe_allow_html=True)

# add an entry message
msg_intro = '<br>A demo using streamlit. It loads an SVM trained model to recognise faces'
msg_intro = msg_intro + ' using Support Vector Machine. The dataset used for this demo is from '
msg_intro = msg_intro + '<span style="color:gray"><b>'+'sklearn datasets</b></span><br><br>'

st.markdown(msg_intro,True)

# The code for showing the model's predictions on the testing set
test_set = st.checkbox('Testing set')
if test_set:
    #st.write('Check the Slidebar to navigate through images in your test set')
    image_index = st.sidebar.number_input('Image index ',0,df.shape[0],0)
    # prepare the image for viewing (return image, and label)
    image_v, label = view_image(image_index)

    # get the predicted label of the image
    label_predicted = str(faces[predictoins[image_index]])

    # show on the app
    st.image(image_v, caption='', use_column_width=False)
    
    # this is just to the message to be printed to the user
    # Notice I add some HTML tags to control color/ bold, etc.. to the msg 
    msg = 'Image '+str(image_index)
    msg = msg+' of <b> <span style="color:blue">'+label+'</span></b>'
    # if correct predictions show predicted value in blue color
    if predictoins[image_index]==labels[image_index]:
        msg = msg+' was classified as '+'<span style="color:blue"><b>'
        msg = msg + label_predicted+ '</b></span>'
    else: # show incorrect predictions in red color
        msg = msg+' was classified as<b> '+'<span style="color:red">'
        msg = msg + label_predicted+ '</span></b>'
    st.markdown(msg,True)

# For uploading new image (i.e. from your computer)
test_new = st.checkbox('Test New Images')
if test_new:
    st.info('Upload a Test Image')
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        features = prepare_image(uploaded_file)
        if st.button('Who is This??'):
            y_hat = model.predict(features)
            st.markdown("He/She is... : "+str(y_hat)+' **'+ faces[int(y_hat)]+'**',True)


# show data frame (actual pixel values)
show_data = st.checkbox('Image Representation')   
# if the show_data is checked then view few rows and columns of the dataset (testing set)             
if show_data: 
    st.write('Recall, each image is represented as a row vector in the dataset of size (width x height). Use the side bar to view more rows/ columns. ')
    # how many rows to show
    n_rows = st.sidebar.slider('Number of Rows to Show',3,df.shape[0])
    # how many columns
    n_cols = st.sidebar.slider('Number of Columns to Show',3,df.shape[1])
    # view n_rows and n_columns of the dataset
    st.dataframe(df.iloc[0:n_rows,0:n_cols])




