# Disaster Response Pipeline
This is the repository for Disaster Response Pipeline project. This project establishes a NLP and Machine Learning pipeline that cleans incoming emergency messages and classifies these messages to 36 categories. It is essential for these messages to be classified correctly as this might affect the society and community's response towards a crisis.

## Table of Contents

  * [Installation](#Installation)
  * [Project Motivation](#Project-Motivation)
  * [File Description](#File-Description)
  * [Results](#Results)
  * [License](#license)

## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*. 

## Project Motivation
I undertook the Data Science Nano Degree at Udacity as I am interested in data science and would like to develop and improve my related technical skills. This repository is the main branch for the Disaster Response Pipeline project. I completed this project as I am interested in developing and improving my pipeline building skills.

## File Descriptions
There are 3 main folders in this repository: app, data and models. App folder contains the neccessary files to run the web application for disaster response project. Data folder contains all the data used in this project and the process_data python file. Created SQL database will also be stored in data folder. Models folder contains python files for train_classifier and will also store the created classifier pickle file. The detailed file distribution are shown below:

 * app
   - template
     * master.html # main page of the web page.
     * go.html # classification result page of the web page
   - run.py # flask file that runs the app.
 * data
   - disaster_categories.csv # data to process
   - disaster_messagies.csv # data to process
   - process_data.py # python file to clean and save cleaned data to a SQL database.
   - DisasterResponse.db # database where cleaned data is stored
 * models
   - train_classifier.py # python files to train and evaluate the model, predicting the messages.
   - classifier.pkl # pickle file that stores the model.
 * README.md


## Results
Depending on the nature of the project, the main findings of the code can be run in terminal and shown in web page.

## Licensing, Authors, and Acknowledgements
For this project, credit must give to Udacity for the data. Any enquiry with the Licensing for the data should directly go to Udacity. Otherwise, feel free to use the code here as you would like! 
