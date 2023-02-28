# User 5 - Policy Writer 
## Use Case 
In a given average ambient temperature what is the estimated cycle life said batteries will reach the end of first/second life?
## Background
Bobby G is a policy writer working in the White House. As the United States shifts towards electrifying its transportation and energy generation sectors, he recognizes the criticality of writing new policy that protects and maintains the newly installed energy infrastructure (i.e., where/when installations should be installed, how often should they be maintained, what is their lifetime, ect.). For the policy to be effective, Bobby G needs to have a solid grasp of how the batteries being used in energy storage units are degrading as a function of time and environmental conditions. Bobby G does not have a background in battery science, but is technically inclined and proficient with statistics. Therefore, Bobby G wants to easily access a database of a particular chemistry battery he knows is going to be installed in a given average ambient temperature, and get an estimate cycle # as to when said batteries will reach the end of second life.

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
- System: Trains ML model on random sample within the dev data frame containing selected features of interest
- System: Trained ML model is tested on the test data frame dataframe containing features of interest to compare 1) capacity retention prediction and/or 2) estimated cycle # at which 80% capacity with #% confidence
