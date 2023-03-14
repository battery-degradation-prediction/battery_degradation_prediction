[![Build Status](https://github.com/battery-degradation-prediction/battery_degradation_prediction/workflows/build/badge.svg)](https://github.com/battery-degradation-prediction/battery_degradation_prediction//actions?query=workflow%3Abuild)
[![Codecov](https://img.shields.io/codecov/c/gh/battery-degradation-prediction/battery_degradation_prediction?token=HYF4KEB84L)](https://codecov.io/gh/battery-degradation-prediction/battery_degradation_prediction)
[![Lint Status](https://github.com/battery-degradation-prediction/battery_degradation_prediction/workflows/lint/badge.svg)](https://github.com/battery-degradation-prediction/battery_degradation_prediction/actions?query=workflow%3Alint)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

BattDegery

# **Forcasting Battery Discharge Capacity Fade**
by Po-Hao Chiu, Anthony Romero, Yi-Shan Lee, Julia Goldblatt

## **Introduction**
<font size = 3>From the aerospace industry, to the automotive industry, to the electrical grid - industries around the world are beginning to transition away from fossil fuels and shift towards greener sources of energy production (e.g., solar, wind, hydro) and utilization. However, in order to effectively distribute/utilize said clean energy for everyday applications, they must be paired with a reliable storage technology, such as secondary (i.e., rechargeable) batteries.

While secondary batteries are capable of achieving desirably high energy and power densities needed for various applications, their lifetime is limited. Over time/use, a batteries capacity degrades. ***Within the literature, there exists a handful of studies dedicated towards furthering our understanding of battery degradation - however to date, no one has accurately been able to forecast/predict the long term "lifetime" of a battery given only limited cycling data.*** If achievable, this would radically improve not only the quality of life for battery manufacturers and consumers, but also provide more reliable estimates as to when maintenance will be needed for given battery powered applications. **Herein, we propose the following models to accurately predict a given batteries future cycle discharge capacity, given only initial/limited cycling data.**
***
## **Getting Started**
***
### How to clone this repository to your local computer 
 1. Open a terminal, and change the current working directory to the desired path you would like to clone our repository to. 

```
$ cd Desired_Path
```
A hypotheical path is provided below for demenstraive purposes, wherein we first create a new directory using `mkdir`, called "Battery_Degredation_Prediction". Next we change to that directroy using `cd` followed by the directory name. Finally, we then print the working directory path using `pwd` (*then  hit enter*) to confirm our desired location:

```
$ mkdir Battery_Degredation_Prediction
```
```
$ cd Battery_Degredation_Prediction
```
```
$ pwd
/home/username/Battery_Degredation_Prediction
```

 2. Now in your desired path, copy and paste the following code below into your command line, and press enter. 
 ```
 $ git clone https://github.com/battery-degradation-prediction/battery_degradation_prediction.git
 ```
3. Success! You have sucessfully cloned the current versions repository to your local computer.
***
## **Requirements to run**
***
### Install and Run Poetry

Poetry **<- link** is a python tool for mananging package dependencies in python, and we utlaize this tool to create a virtual enviroment (containing all required packages) to run our model. Therefore we need to install and run poetry.

#### Installing Poetry

One can install Poetry from the command line by copying the following code below into your terminal, and press enter.
```
$ curl -sSL https://install.python-poetry.org | python3 -
```
#### Running Poetry

Once installed, in order to activate the virtual enviroment, enter the following code in the command line, and press enter.
```
$ poetry shell
```
once executed, you will see a new "path" appear at the start of your command line, that should look identical to whats shown below:
```
(battery-degradation-prediction-py3.10)
```

__Now you are ready to run the model!!__

*Note:* When finsihed, type `exit` in the command line, and hit enter. This will exit poetry's shell (i.e., the vitural enviroment).
```
$ exit
```

### Downloading NASA Li-ion Battery Aging Dataset(s) .mat files
To download the dataset used to train and test our models, visit the link here: **link**. 

*Note, our architecture was designed to be tested on Battery's with a constant current (CC) discharge (i.e., battery #'s 005, 006, 007, 008, 033, 034,and 036) - and is not currently equipped to handle other loading profiles.*

In future versions, you will have the capability to input your own cycling data from common battery cyclers such as Arbin and MACCOR, wherein our software will compile all batteries data into one large dataset, and proceed to train/test the model. 
### Converting .mat files to .csv



where is data going
- notebook exmaple for more clarity


## How to run the model

### Where to place converted .csv files in repository 

### Visulziaing data before and after preprocessing 
 - notebook examples of what is how to 

### 

### 


































# User Stories
new coverage
### Informed Consumer

The User, being a proud owner of a cellular device and somewhat technologically inclined, is looking to purchase a new phone with the longest lasting battery life. The user is proficient at navigating websites, but doesn’t have any kind of technical background in battery science and/or engineering. Therefore, the user is looking for a simple interface that allows them to select a phone model (i.e., battery chemistry), and it will output an estimated “lifetime” (i.e., cycle number) of the phone battery.

 - Interfaces with software
     - Selects Chemistry of phone battery
 - Model outputs predicted cycle life of chemistry

### Researcher

User is a battery researcher, interested in predicting/forecasting the life of a battery given only having a handful of initial cycling data (e.g., given the first 30 cycles, what will be the batteries forecasted 500th cycle look like, and how confident is said prediction?). The User has a strong analytical background and is capable of navigating various software’s, but it not proficient with handling any technicalities/errors/bugs the software may throw at them during use.

- Interfaces with software
 - Input *limited* cycling data into software
     - Reprocess data into dataframe
     - Prompted for features to train model
     - Extract feature(s) 
     - Model is trained on said feature(s)
 - Model outputs predicted cycle life using an *existing* trained fit

### Process/Environmental Engineer

User is an engineer working in the battery field, and wants to know the estimated degradation of certain battery architecture operated at particular conditions to provide their company with 1) a fairly accurate prediction as to when the batteries end of life is, 2) Use said end of life estimate to offer insight into customer warranty of the battery pack(s), and 3) Use said end of life estimate to recommend consumers when/where to recycle their batteries. The User values a simple software interphase, one that is east to use and teach entry level engineers how to operate.

- Interfaces with software
 - Inputs *extensive* cycling data
     - Reprocess data into dataframe
     - Prompted for features to train model
     - Extract feature(s) 
     - Model is trained on said feature(s)
 - Model outputs predicted cycle life on *newly* trained fit

### Database Administrator 

User is proficient in machine learning techniques to predict properties in several domains and wants to strengthen the accuracy of the model's predicted  lifetime of a given battery. They are responsible for accessing/maintaining the actual code/software, debugging any issues, and retraining/finetuning new data when added. Outside of maintaining the main model, this user is also responsible for exploring additional models and integrating them into the existing main branch.

 - Interfaces with software
 - Inputs some sort of Key and password
 - Interphases with code directly

### Policy Writer

User is a policy writer working in the White House. As the United States shifts towards electrifying its transportation and energy generation sectors, they recognize the criticality of writing new policy that protects and maintains the newly adapting energy storage grid infrastructure (i.e., where/when installations should be installed, how often should they be maintained, what is their lifetime, ect.). For the policy to be effective, the user needs to have a solid grasp of how the batteries being used in energy storage units are degrading as a function of time and environmental conditions. The user does not have a background in battery science, but is technically inclined and proficient with statistics. Therefore, user wants to easily access a database of a particular battery chemistry is going to be installed in a given average ambient temperature and get an estimate cycle number as to when said batteries will reach the end of first and second life.

 - Interfaces with software
     - Selects Chemistry battery
     - Prompted for features to train model
     - Extract feature(s) 
     - Model is trained on said feature(s)
 - Model outputs predicted cycle life of chemistry

# Use Cases

### Core Functionality

> **User Inputs and/or accesses a file (i.e. database) of cycling data (containing all data from start-to-end of life) containing a large sample of batteries of a single chemistry, data is processed into python, cleaned for machine learning, and receives a trained ML model that accurately predicts the cycle number at which capacity fades 80% of its original value (ie end of first life)**

### If Time Permitting, Additional Functionality will include

1. After data cleaning, data containing both "raw" and "predicted capacity" will be effectively visualized

2. Using the trained model, user inputs limited amount of cycling data (e.g., first 30 cycles) and software outputs forecasted cycle number at which capacity is expected to fade to 80% 

3. Database Administrator can add a new database for a different chemistry to model 

4. GUI

5. User selects which specific features of interest they wish to train the model on


# Component Specification

**``import_raw_data``**
- receives data from user in some common format (eg excel, matlab) and works data into a python numpy dataframe 

**``data_format_assertion_check``**
- Checks that the imported data contains the necessary minimum features to achieve "high" accuracy
- Checks to see there are only "completed" cycles (no half run cycles)

**``preprocess_raw_data_into_dataframe``**  
- receives data from ``import_raw_data`` and removes any NaN
- additional data generation
- creates a time series within dataframe
- all data received is now formatted to be used for training the model
- returns reprocessed dataframe

**``plot_features``**
- plots data from ``preprocess_raw_data_into_dataframe``
Reprocessed 
**``feature_extraction``**
- receives `` reprocessed_dataframe``, and attempt to reduce dimensionality via of one of the processes: conducts principal component analysis, variational autoencoder, Dynamic mode decomposition

**``model.py``**
- split `` reprocessed_dataframe`` into training and testing datasets
- predicts cycle number at which capacity fades 80% of its original value

**``evalutaion.py``**
- statistical analysis on our predictions
- egs MSE, RMSE
