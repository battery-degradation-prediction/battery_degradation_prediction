
# User Stories

### Informed Consumer

The User, being a proud owner of a cellular device and somewhat technologically inclined, is looking
to purchase a new phone with the the longest lasting battery life. The user is proficient at
navigating websites, but doesn’t have a any kind of technical background in battery science and/or
engineering. Therefore, the user is looking for a simple interface that allows them to select a phone
model(s) (i.e., battery chemistrie(s)), and it will output an estimated “lifetime” (i,e. cycle number)
of the phone battery.

 - Interfaces with software
     - Selects Chemistry of phone battery
 - Model outputs predicted cycle life of chemistry

### Researcher

User is a battery researcher, interested in predicting/forcasting the life of a battery given only having a handful of intial cycling data (e.g., given the first 30 cycles, what will be the batteries forcasted 500th cycle look like, and how confident is said prediction?). The User has a strong analytical background and is capable of navigating various softwares, but it not proficent with handling any technicalilties/errors/bugs the software may throw at them during use. 

 - Interfaces with software
 - Input *limited* cycling data into software
     - Repreocess data into dataframe
     - Prompted for features to train model
     - Extract feature(s) 
     - Model is trained on said feature(s)
 - Model outputs predicted cycle life using an *existing* trained fit

### Process/Environmental Engineer

User is an engineer working in the battery field, and wants to know the estmiated degredation of certain battery architecture operated at partcular conditions to provide their company with 1) a fairly accuracte prediction as to when the batteries end of life is, 2) Use said end of life estimate to offer insight into customer warrenty of the batery pack(s), and 3) Use said end of life estimate to reccomned consumers when/where to recycle their batteries. The User values a simple software interphase, one that is east to use and teach entry level engineers how to operate.

 - Interfaces with software
 - Inputs *extensice* cycling data
     - Repreocess data into dataframe
     - Prompted for features to train model
     - Extract feature(s) 
     - Model is trained on said feature(s)
 - Model outputs predicted cycle life on *newly* trained fit

### Database Administrator 

User is proficeint in machine learning techniques to predict properties in several domains and wants to strengthen the accuracy of the model's predicted  lifetime of a given battery. They are responsible for accesing/maintaining the actual code/software, debugging any issues, and retraining/finetuning new data when added. Outside of maintaining the main model, this user is also repsonsbile for exploring additonal models and integrating them into the existing main branch.

 - Interfaces with software
 - Inputs some sort of Key and password
 - Interphases with code directly

### Policy Writer

User is a policy writer working in the White House. As the United States shifts towards electrifying its transportation and energy generation sectors, they recognize the criticality of writing new policy that protects and maintains the newly adapting energy stroage grid infrastructure (i.e., where/when installations should be installed, how often should they be maintained, what is their lifetime, ect.). For the policy to be effective, the user needs to have a solid grasp of how the batteries being used in energy storage units are degrading as a function of time and environmental conditions. The user does not have a background in battery science, but is technically inclined and proficient with statistics. Therefore, user wants to easily access a database of a particular chemistry battery he knows is going to be installed in a given average ambient temperature, and get an estimate cycle # as to when said batteries will reach the end of first and second life.

 - Interfaces with software
     - Selects Chemistry battery
     - Prompted for features to train model
     - Extract feature(s) 
     - Model is trained on said feature(s)
 - Model outputs predicted cycle life of chemistry

# Use Cases

### Core Functuniality

> **User Inputs and/or accesses a file (i.e. database) of cycling data (containing all data from start-to-end of life) containing a large sample of batteries of a single chemistry, data is proccesed into python, cleaned for machine learning, and recieves a trained ML model that accuarly predicts the cycle number at which capacity fades 80% of its original value (ie end of first life)**

### If Time Permitting, Additonal Functuniality will include

1. After data cleaning, data containing both "raw" and "predicted capcity" will be effectively visualized

2. Using the trained model, user inputs limited amount of cycling data (e.g., first 30 cycles) and software outputs forcasted cycle number at which capacity is expected to fade to 80% 

3. Using the trained model, user inputs limited amount of cycling data (e.g., first 30 cycles) and software outputs forcasted cycle number at which capacity is expected to fade to 30% 

4. GUI

5. User selects features of interest they wish to train the model on


# Component Specification

**``import_raw_data``**
- receives data from user in some common format (eg excel, matlab) and works data into a python numpy dataframe 

**``data_format_assertion_check``**
- Checks that the imported data contains the nescsary minimum features to achieve "high" accuracy
- Checks to see there are only "completed" cycles (no half run cycles)

**``preprocess_raw_data_into_dataframe``**  
- recives data from ``import_raw_data`` and removes any NaN
- additional data generation
- creates a time series within dataframe
- all data received is now formatted to be used for training the model
- returns reporccesed dataframe

**``plot_features``**
- plots data from ``preprocess_raw_data_into_dataframe`` and/or ``cleaned_reprocessed_data`` 

**``cleaned_reprocessed_data``**
- Creates a new dataframe of the users selected features of intertest
- Returns cleaned_reprocessed dataframe

**``feature_extraction``**
- recieves ``cleaned_reprocessed_data``, and attempt to reduce dimensionalality via of one of the processes: conducts principal component analysis, variational autoencoder, Dynamic mode decomposition

**``model.py``**
- recieves lower dimensional data
- split into training and testing datasets
- predicts

**``evalutaion.py``**
- statistical anayslsis on our predictions
- egs MSE, RMSE
