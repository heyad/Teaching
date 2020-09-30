### Face Recognition using SVM

This is a Face Recognition Demo using Support Vector Machines. Interactive features and widgets were created using the great tool [Streamlit](https://www.streamlit.io/). It is an open-source framework, free, Python-based and easy to use tool to build and deploy data-driven and machine learning applications. 

The demo provides three key features 
* It loads the testing set, view it and show you if the predictions of the SVM model was correct or not as can be seen below

![alt text](https://github.com/heyad/Teaching/blob/master/SVMFR/figures/gif.gif "Face Recognition")

* •You can also upload new test images of any size. Currently, it accepts `jpg`, `jpeg`, and `png` formats 

![alt text](https://github.com/heyad/Teaching/blob/master/SVMFR/figures/gifnew.gif "Face Recognition")


This repository contains the following files:

* `streamlitApp.py` this contains the code relevant to streamlit and the functional features 
* `svm_model.pkl` the SVM model
*  `SVM.ipynb`, detailed notebook about SVM and FR
*  `Xy_test` subste of the testing set to run with the demo

#### Requirements 


* You need to have Python installed on your machine 
* The dataset used can be obtained from `sklearn.datasets`
* You need to have streamlit installed, if not, simply issue the following command  

```
$pip install streamlit
```


#### Comments / Questions 

You can reach me at [my staff page](https://www3.rgu.ac.uk/dmstaff/elyan-eyad) or on [linkedin](http://www.linkedin.com/in/elyan )

