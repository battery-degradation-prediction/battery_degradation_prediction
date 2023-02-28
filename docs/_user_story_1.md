# User Story 1 - Informed consumer

## Use Case
which newest phone model will have the longest lasting battery life

## Background
Karen is a proud owner of a cellular device and your average well informed technology enthusiast. Currently, Karen is looking to purchase a new cellular device, and wants to know __which newest phone model will have the longest lasting battery life__ to make the most informed decision. Karen is proficient at navigating websites, but doesn’t have a technical background. Therefore, Karen is looking for a simple interface that allows her to select a phone model(s) (i.e., battery chemistrie(s)), and it will output an estimated “lifetime” of the phone battery.

## Component Design - Describe what the system does?
- User: Selects a battery chemistry
- System: Accesses a database on said chemistry
- System: Processes database files into workable dataframe
- System: Displays what variables/column headers were extracted
- System: Asks user to select features of interest to train model on, and suggests recommendations based on the data extracted from the column headers 
- User: Inputs features of interest
- System: Recognizes Ambient temperature was selected as a feature
- System: Displays all ambient temperatures tested within the dataframe, and prompts user for temperature(s) to train model on
- User: Selects ambient temperature(s) of interest
- System: Reprocess dataframe to work only with users selected  ambient temperature
- System: Tests for various (linear) correlations within dataframe of selected features
- System: Outputs correlations and asks user what to train model on 
- User: selects feature(s) to train model
- System: Double checks user has selected the “best” correlation to train model on. If not, prompts them are they sure they want to continue
- User: If user selected any option but the “best correlation” they will need to “confirm” to continue to train the model 
- __System: Trains ML model on random sample within the dev data frame containing features of interest__
- System: Trained ML model is tested on the test data frame dataframe containing features of interest to compare 1) capacity retention prediction and/or 2) estimated cycle # at which 80% capacity with #% confidence
- Plots