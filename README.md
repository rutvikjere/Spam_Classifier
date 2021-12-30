# Spam_Classifier

Introduction:
This is a throwback project, with an extra edge. A friend and I made a Spam Classifier when we were in our 3rd year of engineering. This project is an inspiration from that project.

The dataset used here is the Enron Email Dataset and I aim to use it to train a Naive Bayes Classifier to the dataset. Further, I deploy the pipeline to a web application to classify whether the input is spam or ham.
One caveat to this dataset is that it is rather small and therefore limits the abilities of the Classifier.

What the files contain:
The Spam_Filter.py file contains the actual preprocessing/training of the dataset and the classifier.
template and static contain the HTML and CSS files.
Finally, the app.py is the file which contains the Flask app and functions which call the other files and the pipeline. This file is run on the command line prompt to launch the web application.

Closing Remarks:
Any criticism or comments are welcome. This was an interesting learning experience as this was the first time I attempted to deploying a machine learning model.

Thank you for checking this repository out!
