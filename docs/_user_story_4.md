# User 4 Story - Technician
## Use Case
Add new data to a given battery chemistry database and retrain model
## Background
Sean is a data scientist who is good using machine learning techniques to predict properties in several domain. To make the model that predicts the lifetime of a battery more robust, __he need to retrain/finetune it when there are new data added into the original dataset. He would be glad if there is a clear direction to help him figuring out where to reach out to new data.__ After the model is trained successfully, he would also like to compare its performance with other models.

## Component Design - Describe what the system does?

- System: Login your account, customer or technician
- User: Choose “technician” and enter account information
- System: Parse account information and match them with technician database.
- System: [Successful]
- System: Display main page, e.g., four boxes directing users to “Model detail”, “Data”, “Training history”, and “Model training”.
- User: Select either category. (For example, click “Model retrain” page).
- System: Directing to “Model training” page.
- System: Select which model architecture wants to be trained, select which dataset is used, and select “Train from scatch” or “Finetune the model”
- User: Click one of the architecture (feed hyperparameters as well), dataset, then choose __finetune the model__.
- System: Check if the model setting matches with any training history. If yes, raise error “This model setting has been trained on {dataset_name} before!”, else: pass.
- System: Read the checkpoint .json file containing the training results from last training process (lr, steps, model params, ..., etc)
- System: Show training progress [===........]
- System: After training, show the performance of the model.
- System: Visualize results, save models, ...,
- User: Click __Choose visualize results__
- System: Launch __plot packages__
- System: What types of figures you want to generate?
- User: Select one option, e.g., loss-iteration, confusion matrix
- System: 