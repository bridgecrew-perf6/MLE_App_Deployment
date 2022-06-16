# Analysis and Deployment of Capstone Project for UCSD Machine Learning Engineering Bootcamp
Nate C. Carnes, PhD

This project aims to predict hospital readmissions (within 30 days) for patients with mental, behavioral, and developmental disorders. The data is from the Healthcare Cost and Utilization Project (HCUP) National Readmission Database (NRD) year 2018; the data will not be shared in this repository because the HCUP Data Use Agreement restricts the sharing of this sensitive database to protect the privacy of patients and partners. If the user obtains access to HCUP data for themselves, the code in this repository can take aggregated and cleaned input data, prepare it, preprocess it, train a DNN model, evaluate the model, and make predictions. HCUP data can be purchased from the HCUP Central Distributor if you wish to work on this project. Otherwise, the saved model in this repository is available for evaluation and prediction.

A DNN model was fit to the training data and yielded binary classification metrics that, despite extensive hyperparameter tuning and model/data refinement, indicated modest but acceptable fit. Nonetheless, this prediction function may be of substantial real-world use for clinicians to assess the likelihood of readmission for their patients. For example, providing clinicians with the accurate and unbiased probability of readmission can help guide treatment, discharge, and follow-up decisions to both improve care and reduce costs. As such, a web app was developed and deployed on streamlit permitting users to easily utilize this prediction function. This repository contains the code and dependencies for this web app. The web app can be accessed at the following link.

https://share.streamlit.io/nateccarnes/mle_app_deployment/main/web_app.py

To use the web app, first follow the link provided above. You can enter patient information on the left-hand-side of the screen (such as their primary diagnosis) using the drop-down menus, and the web app will automatically make a prediction about the likelihood of readmission.

# Instructions

If you clone and download this repository and install the dependencies listed in requirements.txt, you can conduct the data analysis pipeline end-to-end with logging (permitting that you have data) by running the Script.py file. The Model folder contains the saved DNN model for making predictions with your own data (but note that it will need to be aggregated, reduce, and cleaned first). You can also host the web app locally by running the web_app.py file using streamlit.
