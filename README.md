# deep-learning-challenge

## Report on the Neural Network Model


1.	### Overview of the analysis: 
The purpose of the following instructions is to perform data preprocessing and prepare the dataset for machine learning modeling. These steps help ensure that the data is in a suitable format for training and evaluating machine learning models.


    1.1 Identify Target and Feature Variables:
    The first step is to read in the dataset and identify the target variable (the variable you want to predict) and the feature variables (the variables used to make predictions). This step is essential for setting up a supervised machine learning problem.

    1.2	Drop EIN and NAME Columns:
    The EIN and NAME columns are typically identification columns that do not provide useful information for modeling. Dropping them simplifies the dataset.

    1.3	Determine the Number of Unique Values:
    This step helps you understand the uniqueness and diversity of values in each column. It can be useful for identifying categorical variables and potential issues like high cardinality.

    1.4	Count Unique Values:
    For columns with more than 10 unique values, counting the data points for each unique value helps in assessing data distribution and deciding how to handle categorical variables with many categories.

    1.5	Binning Rare Categories:
    When dealing with categorical variables with many unique values, you may want to bin or group infrequent categories into an "Other" category to prevent overfitting and improve model generalization.

    1.6	One-Hot Encoding:
    Using pd.get_dummies(), you can convert categorical variables into binary (0 or 1) columns for each category. This step is essential because most machine learning algorithms require numerical input.

    1.7	Split Data into Features and Target:
    Splitting the data into feature (X) and target (y) arrays is necessary for training and evaluating machine learning models. The train_test_split function is commonly used for this purpose, dividing the data into training and testing datasets.

    1.8	Feature Scaling: 
    Scaling the features helps ensure that numerical features have similar scales, which can improve the performance of many machine learning algorithms. The StandardScaler standardizes the data by removing the mean and scaling to unit variance.
    These preprocessing steps are crucial for getting the data ready for training and testing machine learning models. They improve data quality, handle categorical variables, and ensure that features are in a suitable format for modeling. The specific implementation of these steps may vary depending on the dataset and the machine learning algorithm you plan to use.


2.	### Results: 
In this Section results are discussed
![1st_model](https://github.com/Elee-Saleem/deep-learning-challenge/blob/main/images/1st_model.png)

    2.1	Data Preprocessing:

        	 What variable(s) are the target(s) for your model?

            The target variable for our model is 'IS_SUCCESSFUL.' This variable indicates whether the funding provided by Alphabet Soup was successful in achieving its intended purpose.

        	 What variable(s) are the features for your model?

            The features for our model include all the columns in the dataset except 'IS_SUCCESSFUL,' 'EIN,' and 'NAME.' These features encompass various metadata about the organizations that received funding from Alphabet Soup.

    
        	 What variable(s) should be removed from the input data because they are neither targets nor features?

            The EIN column is typically identification columns that do not provide useful information for modeling. Dropping it simplifies the dataset.


    2.2	Compiling, Training, and Evaluating the Model

        	How many neurons, layers, and activation functions did you select for your neural network model, and why?

            Model 1:
             - Architecture: Three hidden layers with 128, 64, and 16 neurons, respectively.
             - Activation functions: ReLU for hidden layers and sigmoid for the output layer.
             - Loss function: Binary cross-entropy.
             - Optimizer: Adam.
             - Accuracy: Less than 75%
             ![1st_model](https://github.com/Elee-Saleem/deep-learning-challenge/blob/main/images/1st_model.png)

            Model 2:
             - Architecture: Three hidden layers with 512, 128, and 32 neurons, respectively.
             - Activation functions: ReLU for hidden layers and softmax for the output layer (multiclass classification).
             - Loss function: Categorical cross-entropy.
             - Optimizer: Adam.
             - Accuracy: Less than 75%
             ![2nd_model](https://github.com/Elee-Saleem/deep-learning-challenge/blob/main/images/2nd_model.png)

            Model 3:
             - Architecture: Three hidden layers with 128, 64, and 16 neurons, respectively.
             - Activation functions: ReLU for hidden layers and sigmoid for the output layer.
             - Loss function: Binary cross-entropy.
             - Optimizer: Customized Adam optimizer with a learning rate of 0.001.
             - Checkpoint callback: Model weights are saved during training.
             - Accuracy: 75%
             ![3rd_model](https://github.com/Elee-Saleem/deep-learning-challenge/blob/main/images/3rd_model.png)


        	Were you able to achieve the target model performance?
            Yes, I achieved the target model performance.


        	What steps did you take in your attempts to increase model performance?
            i. Included the "NAME" column as an additional feature.
            ii. Augmented the dataset by categorizing the "NAME" column.
            iii. Experimented with different model architectures and hyperparameters.


3.	### Summary: 
The purpose of the analysis is to preprocess the dataset for machine learning modeling. The steps include identifying target and feature variables, dropping unnecessary columns, determining unique values, counting data points, binning rare categories, and performing one-hot encoding. The data is split into features and targets, and feature scaling is applied. The target variable is 'IS_SUCCESSFUL,' while the features include all columns except 'EIN' and 'NAME.' The EIN and NAME columns are removed as they don't provide relevant information. Three neural network models are presented with different architectures and activation functions. Model 1 achieved the target performance of 75%, while Model 2 did not. Model 3 used a customized optimizer and achieved the target performance. Steps to increase model performance included feature engineering, data augmentation, experimenting with architectures and hyperparameters, and customized optimization.


4. #### Describe how you could use a different model to solve the same problem, and explain why you would use that model?
    One alternative model that could be used to solve this problem is a Random Forest Classifier.

    #### Here's how it can be applied:

    #### 4.1 Data Preprocessing:
        Similar to the neural network approach, you would still need to perform data preprocessing steps such as identifying target and feature variables, dropping unnecessary columns (EIN and NAME), determining unique values, handling categorical variables, and splitting the data into features and target.

    #### 4.2 Feature Engineering:
        You may want to perform additional feature engineering, such as creating new features based on domain knowledge or transforming existing features to better represent the underlying relationships.

    #### 4.3 Model Selection - Random Forest:
        Random Forest is a versatile ensemble learning method that works well for both classification and regression problems. It's known for its ability to handle a large number of features and provide insights into feature importance.

    #### 4.4 Model Training and Hyperparameter Tuning:
        You would train the Random Forest model on the preprocessed data. Hyperparameter tuning could be performed to find the best configuration for the Random Forest model. This might involve optimizing parameters like the number of trees, maximum depth of trees, and minimum samples per leaf.

    #### 4.5 Model Evaluation:
        The model's performance would be evaluated using appropriate metrics (e.g., accuracy, precision, recall, F1-score) and, if necessary, cross-validation to ensure robustness.


    #### Why Random Forest:

    #### a. Interpretability:
        Random Forest provides insights into feature importance, which can be crucial in understanding which factors contribute most significantly to the success of funding.

    #### b. Robustness to Overfitting:
        Random Forest is less prone to overfitting compared to complex neural network models, which can be especially beneficial if you have a limited amount of data.

    #### c. Handling Categorical Variables:
        Random Forest naturally handles categorical variables, eliminating the need for one-hot encoding.

    #### d. Training Speed:
        Random Forest models are generally faster to train compared to deep neural networks, especially for large datasets.

    #### e. Ensemble of Decision Trees:
        By aggregating predictions from multiple decision trees, Random Forest can capture complex relationships in the data.
        It's important to note that the choice of model ultimately depends on factors such as the size of the dataset, the nature of the features, computational resources, and the need for interpretability. In this case, a Random Forest could be a strong alternative to the neural network approach, especially if you value interpretability and robustness to overfitting.


## Refrences:
    - Line #19 of "AlphabetSoupCharity.ipynb" file, and line # 33 of "AlphabetSoupCharity_Optimization.ipynb":
        https://machinelearningmastery.com/save-load-keras-deep-learning-models/

    - Chatgpt and other websites helped in answering last question.
