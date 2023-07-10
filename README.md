# Leaf-Iris-Species-Detection-using-DECISION-TREE
ML Python Project
---------------------------------------------------------------------------------------

![](https://github.com/developer-venish/Leaf-Iris-Species-Detection-using-DECISION-TREE/blob/main/demo.png)

---------------------------------------------------------------------------------------

Note :- All the code in this project has been tested and run successfully in Google Colab. I encourage you to try running it in Colab for the best experience and to ensure smooth execution. Happy coding!

---------------------------------------------------------------------------------------

This code performs the following steps:

1. Imports necessary libraries and modules:
   - `load_iris` from `sklearn.datasets` to load the Iris dataset.
   - `pandas` as `pd` for data manipulation and analysis.
   - `numpy` as `np` for numerical computations.
   - `train_test_split` from `sklearn.model_selection` for splitting the dataset into training and testing sets.
   - `DecisionTreeClassifier` from `sklearn.tree` for building a decision tree classifier.
   - `accuracy_score` from `sklearn.metrics` for evaluating the accuracy of the model.
   - `pyplot` from `matplotlib` for creating plots.

2. Loads the Iris dataset using `load_iris()` and assigns it to the variable `dataset`. The dataset contains features in `dataset.data` and corresponding labels in `dataset.target`.

3. Prints the data and target arrays of the dataset.

4. Creates a pandas DataFrame `X` to store the features with column names obtained from `dataset.feature_names`. The labels are stored in `Y`.

5. Splits the dataset into training and testing sets using `train_test_split()` with a test size of 0.25 and a random state of 0. Prints the shapes of the training and testing sets.

6. Initializes an empty list `accuracy` to store the accuracy scores.

7. Trains multiple decision tree classifiers with varying max depths from 1 to 9. For each max depth, the model is fitted on the training data and predictions are made on the testing data. The accuracy score is calculated and appended to the `accuracy` list.

8. Plots a graph showing the accuracy scores against the max depth values.

9. Creates a decision tree classifier with a max depth of 3 and criterion of 'entropy'. The model is fitted on the training data.

10. Makes predictions on the testing data using the trained model and assigns them to `Y_pred`.

11. Concatenates the predicted labels `Y_pred` and the actual labels `Y_test` along the columns and prints the result.

12. Calculates the accuracy of the model using `accuracy_score()` and prints the accuracy percentage.

The code demonstrates loading the Iris dataset, splitting it into training and testing sets, training a decision tree classifier with various max depths, evaluating the accuracy, and making predictions using the trained model.

---------------------------------------------------------------------------------------

A decision tree is a supervised machine learning algorithm that is widely used for classification and regression tasks. It is a flowchart-like structure where each internal node represents a feature, each branch represents a decision rule, and each leaf node represents the outcome or the target variable.

In a decision tree, the goal is to split the dataset based on the most significant attributes that lead to the best separation of the target variable. The splitting process is repeated recursively until a stopping criterion is met, such as reaching a maximum depth or a minimum number of samples per leaf.

For classification tasks, decision trees can be used to create a set of if-else rules that classify the input data into different classes. The decision rules are derived from the training data by maximizing information gain or minimizing impurity measures such as Gini Index or entropy.

For regression tasks, decision trees can be used to predict the continuous target variable by assigning a predicted value to each leaf node based on the average or median of the target variable in that leaf.

Decision trees have several advantages, including interpretability, handling both numerical and categorical data, and handling missing values. However, they can be prone to overfitting if the tree becomes too complex and may not generalize well to unseen data.

Ensemble methods like Random Forest and Gradient Boosting are often used to improve the performance of decision trees by combining multiple decision trees.

---------------------------------------------------------------------------------------
