# Readme

### Symbols in Engineering Drawings

In this tutorial, we will use Random Forest Classifier to build a model that can learn the class of symbols of engineering diagrams.  These symbols appear in engineering diagrams that are called Piping and Instrumentation Diagrams (P & ID), and they are very common in the Oil and Gas industry.  You may want to see our [VIDEO Demo](https://youtu.be/8e1n7mIvACw) on processing and analysing these P&IDs using advance Deep Learning methods! Or refer to our papers if you are interested in knowing more about this problem: 

- E. Elyan, L Jamieson, A. A. Gombe, “Deep learning for symbols detection and classification in engineering drawings”, Neural Networks, 129:91-102, 2020, Elsevier, [DOI 10.1016/j.neunet.2020.05.025](https://doi.org/10.1016/j.neunet.2020.05.025)
- E. Elyan, C.G. Moreno and P. Johnston, “Symbols in Engineering Drawings (SiED): An Imbalanced Dataset Benchmarked by Convolutional Neural Networks”, In 2020 International Joint Conference of the 21st EANN (Engineering Applications of Neural Networks), EANN 2020. Proceedings of the International Neural Networks Society, vol 2. Springer, Cham, [DOI 10.1007/978-3-030-48791-1_16](https://doi.org/10.1007/978-3-030-48791-1_16)

- E. Elyan,C.G. Moreno and C. Jayne, “Symbols classification in engineering drawings”, 2018 International Joint Conference on Neural Networks (IJCNN), Rio de Janeiro, Brazil, 2018, pp. 1-8. [DOI 10.1109/IJCNN.2018.8489087](http://dx.doi.org/10.1109/IJCNN.2018.8489087)

The tutorial will focus on classification of symbols, more specifically:

* You will learn how to load, view and classify instances representing 2D images 
* Create and evaluate A Random Forest Model Classifier (Bagging ensemble-based method) using `sklearn.ensemble.RandomForestClassifie` library. Code is available here as a [Jupyter Notebook](Code)
* Make predictions and visualise the results 
* Use streamlit to create an interactive front end to make predictions and view results. The python file that handles the `streamlit` part is [Here](Code/streamlitApp.py). By the end of the tutorial, you will be able to produce something similar to this [Demo](https://youtu.be/5uHn9IpBmTc)

***Note*** If you are trying to predict a continous value (e.g. regression problem), then you can use `sklearn.ensemble.RandomForestRegressor`