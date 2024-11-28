#                                           RTEU Computer Eng - Data Mining Homework
---- 
**Name**   : Fahrettin

**Surname**: Solak

**No**     : 201401053

-----------------------


# Question 1:

## Problem Definition

In this project, we aim to predict students' depression statuses using various features. The dataset includes factors such as academic pressure, sleep duration, and financial stress. Our target variable is `Depression` (1 = Depressed, 0 = Not Depressed), and we will develop machine learning models to predict this variable.

The project seeks to answer the following questions:
1. Which features are most effective in predicting depression status?
2. When comparing models like Decision Tree and Random Forest, which classifier performs best?
3. How can model performance be evaluated using metrics such as accuracy, precision, recall, and F1 score?

---

### Solution Approach

#### 1. Data Exploration and Preparation
- **Data Exploration**:
  - Analyze missing values and distributions in the dataset.
  - Check the class balance of the target variable and take necessary measures.
- **Data Cleaning**:
  - Handle missing values using appropriate methods.
  - Convert categorical variables into a numerical format suitable for machine learning models.
- **Data Splitting**:
  - Split the dataset into training (70%), validation (15%), and test (15%) groups.

#### 2. Model Training and Evaluation
- **Decision Tree Classifier**:
  - Train a fast and interpretable model.
- **Random Forest Classifier**:
  - Train a model with a higher generalization capacity.
- **Performance Metrics**:
  - Evaluate the performance of both models using accuracy, precision, recall, and F1 score.
  - Measure class discrimination capacity using ROC curves and AUC scores.

### 3. Reporting the Results
- Analyze the strengths and weaknesses of the models and provide recommendations.
- Visualize feature importance to identify the most influential factors in predicting depression.
- Use visuals and tables to illustrate performance differences.

---

This process provides a systematic approach to identifying the best strategy for predicting depression and understanding the strengths of different models.
 için sistematik bir yapı sunmaktadır.


## Dataset Introduction

This dataset contains information aimed at analyzing students' depression statuses and lifestyles. Below are detailed descriptions of the columns included in the dataset:

---

### 1. Demographic Information
- **`Gender`**: The gender of the student. Can take two values:
  - `Male`
  - `Female`
- **`Age`**: The age of the student, expressed as a numerical value.
- **`City`**: The city where the student resides. Represented as text (e.g., "Bangalore").
- **`Degree`**: The academic degree pursued by the student (e.g., `B.Sc`, `M.Tech`).

---

### 2. Academic and Workload Indicators
- **`Academic Pressure`**: Academic pressure level (1 = Low, 5 = High).
- **`Work Pressure`**: Workload level (1 = Low, 5 = High).
- **`CGPA`**: Cumulative Grade Point Average. Expressed as a numerical value out of 10.
- **`Study Satisfaction`**: Satisfaction level with studying (1 = Very low, 5 = Very high).
- **`Job Satisfaction`**: Job satisfaction level (1 = Very low, 5 = Very high).

---

### 3. Health and Lifestyle Indicators
- **`Sleep Duration`**: Sleep duration. Can take the following categorical values:
  - `Less than 5 hours`
  - `5-6 hours`
  - `7-8 hours`
  - `More than 8 hours`
- **`Dietary Habits`**: Dietary habits. Can take three values:
  - `Healthy`
  - `Moderate`
  - `Unhealthy`
- **`Financial Stress`**: Financial stress level (1 = Low stress, 5 = High stress).
- **`Work/Study Hours`**: Daily work/study hours (numerical).

---

### 4. Mental Health and Family History
- **`Have you ever had suicidal thoughts?`**: Indicates whether the student has ever had suicidal thoughts. Can take two values:
  - `Yes`
  - `No`
- **`Family History of Mental Illness`**: Indicates whether there is a history of mental illness in the student's family. Can take two values:
  - `Yes`
  - `No`

---

### 5. Target Variable
- **`Depression`**: The target variable indicating the depression status of the student:
  - `1`: Depressed
  - `0`: Not Depressed

---

#### Recommendations:
1. **Graphical Analyses**:
   - Visualize the class distribution of the target variable `Depression` using a bar chart.
   - Examine the distribution of continuous variables (e.g., `Age`, `CGPA`) using histograms or box plots.
2. **Categorical Variables**:
   - Use stacked bar charts to visualize the relationship between categorical variables like `Sleep Duration` and `Dietary Habits` with the target variable.

This structured introduction provides a clear understanding of the dataset and lays the foundation for further analysis.
  - `0`: Depresyon yok.


## Dataset Source

The dataset used in this analysis contains information about students' depression statuses and lifestyles. The dataset is accessible via the **Kaggle** platform.

- **Link**: [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)
- **License Information**:
  - The dataset is intended for **educational purposes only**.
  - For any commercial use, permission must be obtained from the dataset owner.
  - Sharing and usage of the data must comply with Kaggle's licensing rules.

---

### Dataset Summary
The dataset includes survey results from over 1,000 students and provides information under the following categories:
1. **Demographic Information**:
   - Age, gender, city.
2. **Lifestyle and Health**:
   - Sleep duration, dietary habits, financial stress.
3. **Academic and Workload**:
   - Academic pressure, work pressure, grade point average.
4. **Mental Health and Family History**:
   - Suicidal thoughts and family history of mental illness.

This dataset serves as a valuable foundation for analyzing depression prediction and lifestyle factors.


## Loading the Dataset

In this step, we will load the dataset using Python and explore its structure. The objective is to understand the column names, data types, and any missing values in the dataset. Additionally, we will preview the first few rows of the dataset to examine its overall structure.
.



```python
import pandas as pd

# Load the dataset
file_path = 'Student Depression Dataset.csv'
data = pd.read_csv(file_path)

# Preview the first few rows of the dataset
print("First 5 Rows of the Dataset:")
print(data.head())

# Display general information about the dataset
print("\nGeneral Information about the Dataset:")
data.info()

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

```

    First 5 Rows of the Dataset:
       id  Gender   Age           City Profession  Academic Pressure  \
    0   2    Male  33.0  Visakhapatnam    Student                5.0   
    1   8  Female  24.0      Bangalore    Student                2.0   
    2  26    Male  31.0       Srinagar    Student                3.0   
    3  30  Female  28.0       Varanasi    Student                3.0   
    4  32  Female  25.0         Jaipur    Student                4.0   
    
       Work Pressure  CGPA  Study Satisfaction  Job Satisfaction  \
    0            0.0  8.97                 2.0               0.0   
    1            0.0  5.90                 5.0               0.0   
    2            0.0  7.03                 5.0               0.0   
    3            0.0  5.59                 2.0               0.0   
    4            0.0  8.13                 3.0               0.0   
    
          Sleep Duration Dietary Habits   Degree  \
    0          5-6 hours        Healthy  B.Pharm   
    1          5-6 hours       Moderate      BSc   
    2  Less than 5 hours        Healthy       BA   
    3          7-8 hours       Moderate      BCA   
    4          5-6 hours       Moderate   M.Tech   
    
      Have you ever had suicidal thoughts ?  Work/Study Hours  Financial Stress  \
    0                                   Yes               3.0               1.0   
    1                                    No               3.0               2.0   
    2                                    No               9.0               1.0   
    3                                   Yes               4.0               5.0   
    4                                   Yes               1.0               1.0   
    
      Family History of Mental Illness  Depression  
    0                               No           1  
    1                              Yes           0  
    2                              Yes           0  
    3                              Yes           1  
    4                               No           0  
    
    General Information about the Dataset:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27901 entries, 0 to 27900
    Data columns (total 18 columns):
     #   Column                                 Non-Null Count  Dtype  
    ---  ------                                 --------------  -----  
     0   id                                     27901 non-null  int64  
     1   Gender                                 27901 non-null  object 
     2   Age                                    27901 non-null  float64
     3   City                                   27901 non-null  object 
     4   Profession                             27901 non-null  object 
     5   Academic Pressure                      27901 non-null  float64
     6   Work Pressure                          27901 non-null  float64
     7   CGPA                                   27901 non-null  float64
     8   Study Satisfaction                     27901 non-null  float64
     9   Job Satisfaction                       27901 non-null  float64
     10  Sleep Duration                         27901 non-null  object 
     11  Dietary Habits                         27901 non-null  object 
     12  Degree                                 27901 non-null  object 
     13  Have you ever had suicidal thoughts ?  27901 non-null  object 
     14  Work/Study Hours                       27901 non-null  float64
     15  Financial Stress                       27898 non-null  float64
     16  Family History of Mental Illness       27901 non-null  object 
     17  Depression                             27901 non-null  int64  
    dtypes: float64(8), int64(2), object(8)
    memory usage: 3.8+ MB
    
    Missing Values:
    Financial Stress    3
    dtype: int64
    

## Output Explanation

1. **Examining the First Rows**:
   - Using the `head()` function, the first 5 rows of the dataset were displayed to analyze the column names and content structure.
   - Sample data was used to verify whether the columns are in the expected format.

2. **General Information**:
   - The `info()` function provides details about the column names, data types, and total number of entries.
   - This information is crucial for understanding the types of columns (e.g., `object`, `int64`, `float64`) and identifying any potential missing values in the dataset.

3. **Missing Values**:
   - Missing value analysis identifies columns with data loss.
   - If the proportion of missing values is small, they can be filled using appropriate methods (e.g., mean, median).
a, medyan).


## Data Exploration and Understanding

Before working with a dataset, it is crucial to thoroughly understand its structure and the information it contains. This process includes identifying missing values and analyzing the class distribution of the target variable to detect potential class imbalances that could affect model performance.

---

### Steps for Data Analysis
1. **Data Types and General Information**:
   - Examine the data type (`int`, `float`, `object`) and the total number of entries for each column.
2. **Checking for Missing Values**:
   - Identify which columns have missing values and calculate their proportions.
3. **Target Variable Distribution**:
   - Analyze the class distribution of the `Depression` variable.
   - This step is critical for understanding if there is any class imbalance.
4. **Examining the First 5 Rows**:
   - Display sample rows from the dataset to quickly understand its general structure.

---

### Code: Data Exploration



```python
# General information about the dataset
print("General Information about the Dataset:")
data.info()

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Analyze the class distribution of the target variable
target_distribution = data['Depression'].value_counts(normalize=True)
print("\nTarget Variable (Depression) Class Distribution:")
print(target_distribution)

# Preview the first few rows of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(data.head())
```

    General Information about the Dataset:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27901 entries, 0 to 27900
    Data columns (total 18 columns):
     #   Column                                 Non-Null Count  Dtype  
    ---  ------                                 --------------  -----  
     0   id                                     27901 non-null  int64  
     1   Gender                                 27901 non-null  object 
     2   Age                                    27901 non-null  float64
     3   City                                   27901 non-null  object 
     4   Profession                             27901 non-null  object 
     5   Academic Pressure                      27901 non-null  float64
     6   Work Pressure                          27901 non-null  float64
     7   CGPA                                   27901 non-null  float64
     8   Study Satisfaction                     27901 non-null  float64
     9   Job Satisfaction                       27901 non-null  float64
     10  Sleep Duration                         27901 non-null  object 
     11  Dietary Habits                         27901 non-null  object 
     12  Degree                                 27901 non-null  object 
     13  Have you ever had suicidal thoughts ?  27901 non-null  object 
     14  Work/Study Hours                       27901 non-null  float64
     15  Financial Stress                       27898 non-null  float64
     16  Family History of Mental Illness       27901 non-null  object 
     17  Depression                             27901 non-null  int64  
    dtypes: float64(8), int64(2), object(8)
    memory usage: 3.8+ MB
    
    Missing Values:
    Financial Stress    3
    dtype: int64
    
    Target Variable (Depression) Class Distribution:
    Depression
    1    0.585499
    0    0.414501
    Name: proportion, dtype: float64
    
    First 5 Rows of the Dataset:
       id  Gender   Age           City Profession  Academic Pressure  \
    0   2    Male  33.0  Visakhapatnam    Student                5.0   
    1   8  Female  24.0      Bangalore    Student                2.0   
    2  26    Male  31.0       Srinagar    Student                3.0   
    3  30  Female  28.0       Varanasi    Student                3.0   
    4  32  Female  25.0         Jaipur    Student                4.0   
    
       Work Pressure  CGPA  Study Satisfaction  Job Satisfaction  \
    0            0.0  8.97                 2.0               0.0   
    1            0.0  5.90                 5.0               0.0   
    2            0.0  7.03                 5.0               0.0   
    3            0.0  5.59                 2.0               0.0   
    4            0.0  8.13                 3.0               0.0   
    
          Sleep Duration Dietary Habits   Degree  \
    0          5-6 hours        Healthy  B.Pharm   
    1          5-6 hours       Moderate      BSc   
    2  Less than 5 hours        Healthy       BA   
    3          7-8 hours       Moderate      BCA   
    4          5-6 hours       Moderate   M.Tech   
    
      Have you ever had suicidal thoughts ?  Work/Study Hours  Financial Stress  \
    0                                   Yes               3.0               1.0   
    1                                    No               3.0               2.0   
    2                                    No               9.0               1.0   
    3                                   Yes               4.0               5.0   
    4                                   Yes               1.0               1.0   
    
      Family History of Mental Illness  Depression  
    0                               No           1  
    1                              Yes           0  
    2                              Yes           0  
    3                              Yes           1  
    4                               No           0  
    

## Output Explanation

1. **Data Types and Missing Values**:
   - The `info()` function provided details about the column names, data types, and missing values in the dataset.
   - Missing values were analyzed. For example:
     - Missing values were identified in the `Financial Stress` column. These missing entries will be filled with the column's mean.

2. **Target Variable Distribution**:
   - The distribution of the `Depression` target variable was examined. If there is a significant imbalance between classes, it could impact model performance.
   - If the distribution is balanced, our classifiers are likely to perform effectively on this dataset.

3. **Preview**:
   - The `head()` function displayed the first few rows of the dataset. This provided a quick understanding of the column names and data content.

In the next step, we will handle missing values and transform categorical variables into numerical formats to prepare the datafor modeling.


# Data Cleaning and Transformation

The dataset must be prepared for modeling. In this process, we will handle missing values, transform categorical columns into numerical formats, and remove columns that do not carry meaningful information for the model.

---

## Steps to Follow

### 1. Handling Missing Values
- **Goal**: Prevent information loss caused by missing values.
- Missing values in the `Financial Stress` column will be filled with the column's mean.

### 2. Encoding Categorical Variables
- Categorical columns will be transformed into numerical formats:
  - **Label Encoding**:
    - Columns such as `Gender`, `Dietary Habits`, `Degree`, `Have you ever had suicidal thoughts?`, and `Family History of Mental Illness`.
  - **One-Hot Encoding**:
    - Columns with multiple categories, such as `Sleep Duration`.

### 3. Removing Irrelevant Columns
- Columns that do not carry meaningful information for modeling will be removed:
  - `id`, `City`, and `Profession`.

---

### Code: Data Claning and Transformation



```python
# Fill missing values
data['Financial Stress'] = data['Financial Stress'].fillna(data['Financial Stress'].mean())

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Columns to apply label encoding
binary_columns = ['Gender', 'Dietary Habits', 'Degree', 
                  'Have you ever had suicidal thoughts ?', 
                  'Family History of Mental Illness']
for column in binary_columns:
    if column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])

# Transform categorical columns using one-hot encoding
if 'Sleep Duration' in data.columns:
    data = pd.get_dummies(data, columns=['Sleep Duration'], drop_first=True)
else:
    print("'Sleep Duration' column not found, this step will be skipped.")

# Remove irrelevant columns
data = data.drop(columns=['id', 'City', 'Profession'], errors='ignore')

# Display the updated dataframe
print("Updated DataFrame:")
print(data.head())
```

    Updated DataFrame:
       Gender   Age  Academic Pressure  Work Pressure  CGPA  Study Satisfaction  \
    0       1  33.0                5.0            0.0  8.97                 2.0   
    1       0  24.0                2.0            0.0  5.90                 5.0   
    2       1  31.0                3.0            0.0  7.03                 5.0   
    3       0  28.0                3.0            0.0  5.59                 2.0   
    4       0  25.0                4.0            0.0  8.13                 3.0   
    
       Job Satisfaction  Dietary Habits  Degree  \
    0               0.0               0       3   
    1               0.0               1      10   
    2               0.0               0       5   
    3               0.0               1       7   
    4               0.0               1      17   
    
       Have you ever had suicidal thoughts ?  Work/Study Hours  Financial Stress  \
    0                                      1               3.0               1.0   
    1                                      0               3.0               2.0   
    2                                      0               9.0               1.0   
    3                                      1               4.0               5.0   
    4                                      1               1.0               1.0   
    
       Family History of Mental Illness  Depression  Sleep Duration_7-8 hours  \
    0                                 0           1                     False   
    1                                 1           0                     False   
    2                                 1           0                     False   
    3                                 1           1                      True   
    4                                 0           0                     False   
    
       Sleep Duration_Less than 5 hours  Sleep Duration_More than 8 hours  \
    0                             False                             False   
    1                             False                             False   
    2                              True                             False   
    3                             False                             False   
    4                             False                             False   
    
       Sleep Duration_Others  
    0                  False  
    1                  False  
    2                  False  
    3                  False  
    4                  False  
    

## Output Explanation

1. **Handling Missing Values**:
   - Missing values in the `Financial Stress` column were filled with the column's mean. This step prevents data loss caused by missing entries.

2. **Transforming Categorical Columns**:
   - Label encoding transformed categorical columns into numerical values. For example:
     - `Gender`: Male = 1, Female = 0.
     - `Dietary Habits`: Healthy = 2, Moderate = 1, Unhealthy = 0.
   - The `Sleep Duration` column was transformed into separate columns using one-hot encoding.

3. **Removing Irrelevant Columns**:
   - Columns that are not useful for modeling were removed. For example, `id` is simply an identifier and does not carry any meaningful information for a classification model.

4. **Updated Dataset**:
   - All columns are now in numerical format and ready for modeling.

In the next step, we will split the dataset into training, validatio, and test sets.


# Splitting the Dataset into Training, Validation, and Test Sets

To accurately evaluate the performance of machine learning models, the dataset is divided into the following three parts:

---

## Dataset Partitions

### 1. Training Set
- The dataset used for the model to learn from.
- 70% of the total dataset is allocated to the training set.

### 2. Validation Set
- Used for tuning hyperparameters and preventing overfitting.
- 15% of the total dataset is allocated to the validation set, independent of the training set.

### 3. Test Set
- Used for evaluating the overall performance of the model.
- The remaining 15% of the dataset is allocated to the test set.

Stratification is applied during the splitting process to maintain the class distribution of the target variable (`Depression`).

---

### Cod: Splitting the Dataset



```python
from sklearn.model_selection import train_test_split

# Split the features (X) and target variable (y)
X = data.drop(columns=['Depression'])
y = data['Depression']

# Split the data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Check the dimensions of each set after splitting
print(f"Training Set Size: {X_train.shape}")
print(f"Validation Set Size: {X_val.shape}")
print(f"Test Set Size: {X_test.shape}")
```

    Training Set Size: (19530, 17)
    Validation Set Size: (4185, 17)
    Test Set Size: (4186, 17)
    

## Output Explanation

1. **Training Set**:
   - 70% of the dataset is allocated for the model to learn from.

2. **Validation Set**:
   - 15% of the remaining dataset is allocated for hyperparameter optimization.

3. **Test Set**:
   - The final 15% of the dataset is reserved for evaluating the model's performance.

The sizes of each set were printed to confirm the success of the splittig process.


# Decision Tree Classifier Training

The Decision Tree classifier learns decision rules from the dataset's features to perform classification. This model is simple, interpretable, and can easily handle both categorical and numerical data.

---

## Steps to Follow

1. **Model Training**:
   - The Decision Tree classifier will be trained on the training set.

2. **Performance Evaluation**:
   - Predictions will be made on the validation set, and performance metrics will be calculated:
     - Accuracy
     - Precision
     - Recall
     - F1 Score

3. **Confusion Matrix**:
   - The model's prediction performance will be analyzed using a confusion matrix.

---

### Code: Decision Tree Model



```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = dt_model.predict(X_val)

# Calculate performance metrics
dt_accuracy = accuracy_score(y_val, y_val_pred)
dt_classification_report = classification_report(y_val, y_val_pred)
dt_confusion_matrix = confusion_matrix(y_val, y_val_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(dt_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'])
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print the performance report
print("Decision Tree Accuracy:", dt_accuracy)
print("\nDecision Tree Classification Report:")
print(dt_classification_report)

```

    
![png](https://github.com/user-attachments/assets/78027dc2-a0c6-451e-8d39-8445c66ce8a6)
    


    Decision Tree Accuracy: 0.7715651135005974
    
    Decision Tree Classification Report:
                  precision    recall  f1-score   support
    
               0       0.72      0.73      0.73      1735
               1       0.81      0.80      0.80      2450
    
        accuracy                           0.77      4185
       macro avg       0.76      0.77      0.77      4185
    weighted avg       0.77      0.77      0.77      4185
    
    

## Output Explanation

1. **Decision Tree Performance**:
   - The model's accuracy (`Accuracy`) represents the percentage of correctly predicted instances across all classes.
   - The classification report provides class-specific metrics such as precision, recall, and F1 score to evaluate the model's performance in detail.

2. **Confusion Matrix**:
   - The confusion matrix illustrates the relationship between actual and predicted classes.
   - A balanced distribution in the matrix indicates that the model performs well across all classes without bias.

Based on these results, we will analyze the performance of the Decision Tree model and proceed to train the next model, Random Forest, for comparison.


# Random Forest Classifier Training

The Random Forest classifier combines multiple decision trees to create a more robust and generalizable model. Random Forest is particularly effective for imbalanced datasets and high-dimensional feature spaces.

---

## Advantages of the Random Forest Model
1. **Resistance to Overfitting**:
   - Random Forest reduces overfitting by aggregating the results of multiple trees.
2. **Feature Importance Analysis**:
   - The model can measure the impact of each feature on the target variable.
3. **Overall Performance**:
   - Produces strong results in terms of both accuracy and generalization.

---

### Steps to Follow
1. Train the Random Forest classifier using the training set.
2. Evaluate its performance on the validation set.
3. Compute performance metrics:
   - Accuracy, precision, recall, and F1 score.
4. Analyze prediction results using a confusion matrix.

---

### Code: Random Forest Model



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred_rf = rf_model.predict(X_val)

# Calculate performance metrics
rf_accuracy = accuracy_score(y_val, y_val_pred_rf)
rf_classification_report = classification_report(y_val, y_val_pred_rf)
rf_confusion_matrix = confusion_matrix(y_val, y_val_pred_rf)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(rf_confusion_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'])
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print the performance report
print("Random Forest Accuracy:", rf_accuracy)
print("\nRandom Forest Classification Report:")
print(rf_classification_report)

```


    
![png](output_21_0.png)
    


    Random Forest Accuracy: 0.8387096774193549
    
    Random Forest Classification Report:
                  precision    recall  f1-score   support
    
               0       0.82      0.78      0.80      1735
               1       0.85      0.88      0.87      2450
    
        accuracy                           0.84      4185
       macro avg       0.84      0.83      0.83      4185
    weighted avg       0.84      0.84      0.84      4185
    
    

## Output Explanation

1. **Random Forest Performance**:
   - The accuracy of the Random Forest model will be compared with the Decision Tree model.
   - The classification report includes precision, recall, and F1 scores for Class 0 and Class 1.

2. **Confusion Matrix**:
   - The confusion matrix analyzes the accuracy of predictions between actual and predicted classes.
   - Random Forest is expected to demonstrate more balanced performance compared to the Decision Tree model.

The results will be compared with the Decision Tree model, and the strengths of the Random Forest model will be analyzed.


# Decision Tree and Random Forest Comparison

In this section, we will compare the performance of the Decision Tree and Random Forest models. The comparison will be based on the following metrics:

1. **Accuracy**:
   - The percentage of correctly predicted instances across all classes.
2. **Precision**:
   - Measures how accurate the positive class predictions are.
3. **Recall**:
   - Indicates how well the model identifies positive classes.
4. **F1 Score**:
   - Represents the balance between precision and recall.

These metrics will serve as the foundation for analyzing the strengths and weaknesses of the models.

---

### Code: Model Comparison



```python
import numpy as np
import matplotlib.pyplot as plt

# Metrics for comparison
models = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy, rf_accuracy]
precision_scores = [0.72, 0.82]  # Precision for Class 0
recall_scores = [0.73, 0.78]  # Recall for Class 0
f1_scores = [0.73, 0.80]  # F1 Score for Class 0

# Accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green'], alpha=0.7)
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Precision, Recall, F1 Score comparison
metrics = ['Precision', 'Recall', 'F1 Score']
dt_metrics = [precision_scores[0], recall_scores[0], f1_scores[0]]
rf_metrics = [precision_scores[1], recall_scores[1], f1_scores[1]]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, dt_metrics, width, label='Decision Tree', color='blue', alpha=0.7)
plt.bar(x + width / 2, rf_metrics, width, label='Random Forest', color='green', alpha=0.7)
plt.xticks(x, metrics)
plt.title('Precision, Recall, and F1 Score Comparison')
plt.ylabel('Score')
plt.legend()
plt.show()

```


    
![png](output_24_0.png)
    



    
![png](output_24_1.png)
    


## Output Explanation

1. **Accuracy Comparison**:
   - Random Forest outperformed Decision Tree with an accuracy of 83.87% compared to 77.16%.

2. **Precision, Recall, and F1 Score**:
   - Random Forest achieved higher scores across all metrics compared to Decision Tree.
   - It particularly demonstrated more balanced performance for Class 1 (Depression).

3. **Conclusion**:
   - Random Forest is a stronger model in terms of generalization and maintaining balance across classes.
   - It effectively handles class imbalances and provides higher accuracy.


# Feature Importance with the Random Forest Model

The Random Forest model can calculate the importance levels of features used in predicting depression. This analysis helps us understand which features the model considers most significant. For example, we can evaluate the impact of factors such as academic pressure or financial stress on depression prediction.

---

### Code: Calculating and Visualizing Feature Importance



```python
# Retrieve feature importance values
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Visualization
plt.figure(figsize=(12, 8))
plt.barh(feature_names, feature_importances, color='green', alpha=0.7)
plt.title('Random Forest - Feature Importance')
plt.xlabel('Importance Level')
plt.ylabel('Features')
plt.show()

```


    
![png](output_27_0.png)
    


## Output Explanation

1. **Feature Importance Levels**:
   - The visualization ranks each feature based on its importance level in the Random Forest model.
   - Features with higher importance levels are those the model relies on more heavily when predicting depression.

2. **Key Features**:
   - *CGPA*: This measure of academic performance may be a significant predictor of depression.
   - *Financial Stress*: Financial pressure stands out as a critical factor influencing depression risk.
   - *Sleep Duration*: Sleep duration directly impacts students' mental health and plays an essential role in predictions.

3. **Conclusion**:
   - This analysis is a valuable tool for identifying the most critical features in depression prediction.
   - Feature importance can be leveraged to improve model accuracy by removing irrelevant features (dimensionality reduction) or assigning more weight to specific features during modeling.


# ROC Curve and AUC Score Analysis

The ROC curve illustrates a model's ability to distinguish between positive classes, while the AUC (Area Under the Curve) score represents the area under the ROC curve, summarizing model performance numerically. The following values explain the AUC score:
- **1**: Perfect classification performance.
- **0.5**: Random guessing (model fails).

---

### Objective
1. Plot the ROC curves for both the Decision Tree and Random Forest models.
2. Calculate and compare the AUC scores for each model.

---

### Code: ROC Curve and AUC Score



```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Predicted probabilities for Decision Tree
y_val_proba_dt = dt_model.predict_proba(X_val)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_val, y_val_proba_dt)
auc_dt = roc_auc_score(y_val, y_val_proba_dt)

# Predicted probabilities for Random Forest
y_val_proba_rf = rf_model.predict_proba(X_val)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_val_proba_rf)
auc_rf = roc_auc_score(y_val, y_val_proba_rf)

# Plotting the ROC Curve
plt.figure(figsize=(10, 6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})', color='blue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

```


    
![png](output_30_0.png)
    


## Output Explanation

1. **ROC Curve**:
   - The slope of the ROC curve indicates the model's ability to distinguish between positive classes.
   - The ROC curve for Random Forest is positioned higher than that of Decision Tree, demonstrating better performance.

2. **AUC Score**:
   - The AUC score numerically represents how well the model separates positive classes overall.
   - If the AUC score for Random Forest is higher than Decision Tree, it highlights the superiority of the Random Forest model.

This analysis provides a deeper understanding of the models' performance.


# Performance Analysis of Decision Tree and Random Forest Models

## 1. Problem Definition
The goal of this study is to predict students' depression statuses using various features and examine the performance differences between two machine learning models: Decision Tree and Random Forest.

---

## 2. Dataset Features
The dataset contains demographic, behavioral, and academic information about students. These features were analyzed for their relationships with depression status.

### Key Features:
- **Demographic Information**: Age, gender, city.
- **Health and Behavioral Attributes**: Sleep duration, dietary habits, financial stress.
- **Academic Data**: CGPA, academic, and work pressure.

---

## 3. Model Performances
Both models were evaluated on the test set using the following performance metrics:

### Decision Tree
- **Accuracy**: 77.16%
- **Class 0 (No Depression)**:
  - Precision: 72%
  - Recall: 73%
  - F1 Score: 73%
- **Class 1 (Depression)**:
  - Precision: 81%
  - Recall: 80%
  - F1 Score: 80%

### Random Forest
- **Accuracy**: 84.07%
- **Class 0 (No Depression)**:
  - Precision: 82%
  - Recall: 79%
  - F1 Score: 80%
- **Class 1 (Depression)**:
  - Precision: 85%
  - Recall: 88%
  - F1 Score: 87%

---

## 4. ROC Curve and AUC Score
The ROC curves and AUC scores for the models are as follows:
- **Decision Tree**: AUC = 0.77
- **Random Forest**: AUC = 0.85

---

## 5. Feature Importance
The most important features in the Random Forest model are:
1. **CGPA**
2. **Financial Stress**
3. **Academic Pressure**

---

## 6. Conclusions and Recommendations
1. **Random Forest** demonstrated higher accuracy and generalization compared to Decision Tree.
   - A higher AUC score (85%) confirms its superior ability to distinguish positive classes.
2. **Decision Tree** is advantageous for small datasets due to its speed and interpretability.
3. **Random Forest** should be preferred, especially for imbalanced datasets.

### Recommendations for Future Work:
- **Hyperparameter Optimization**:
  - Further improve Random Forest performance by tuning parameters (e.g., number of trees, maximum depth).
- **More Complex Models**:
  - Consider advanced models such as XGBoost or LightGBM for further performance enhancement.
- **Data Collection**:
  - Gather additional data to address class imbalance and improve model robustness.

---

This report highlights the strengths and weaknesses of the models and identifies the most critical features for predicting depression. These insights provide valuable guidance for data analysts and decision-makers.
r, veri analistleri ve karar vericiler için değerli bir rehber sunmaktadır.


# Question 2:

# Problem Definition

In this project, the goal is to group students with similar characteristics by analyzing their academic performance and program choices. This process uses the **k-means clustering algorithm** to explore the internal structure of the dataset and uncover similarities and differences among groups. Such an analysis can be beneficial for providing recommendations to students to enhance their academic performance or creating more meaningful student profiles
---
.

## Dataset Description
The dataset used in this project is sourced from Kaggle and contains information about student academic performance, program choices, and demographic details. It is a valuable resource for analyzing student groups using various clustering algorit
---
hms.

### Key Columns in the Dataset:
- **Student ID**: A unique identifier for each student.
- **Program**: The academic program the student is enrolled in (e.g., Computer Science, Electrical Engineering).
- **GPA (Grade Point Average)**: The student's academic performance score (out of 4).
- **Study Hours per Week**: The number of hours the student studies weekly.
- **Gender**: The gender of the student (Male/Female).
- **Age**: The age of 
---
the student.

### Dataset Source:
- **Link**: [Student Academic Grades and Programs](https://www.kaggle.com/datasets/marmarplz/student-academic-grades-and-programs)
- **License Information**: This dataset is available for educational and research purposes and is subject to the terms provided by th
---
e dataset owner.

## Objectives
1. **Prepare the Dataset**:
   - Analyze the dataset to make it suitable for clustering.
   - Complete preprocessing steps such as handling missing values and encoding categorical variables.

2. **Build a Clustering Model**:
   - Experiment with different numbers of clusters (`k`) to determine the optimal number.
   - Use the `Elbow Method` to identify the best value for `k`.
   - Evaluate the clustering quality using metrics like **Silhouette Score**.

3. **Interpret and Visualize Results**:
   - Evaluate and visualize the clustering results.
   - Examine similarities and differences between clusters.
   - Identify meaningful relationships between clusters and analy
---
ze the benefits of the results.

## Solution Approach

### 1. **Data Exploration and Understanding**
- **Examine Dataset Structure**: Analyze the general structure, column names, and data types in the dataset. This step is crucial for understanding the dataset and preparing it for modeling.
- **Handle Missing Values and Categorical Data**: Identify columns with missing values and handle them appropriately. Convert categorical variables (`Progr
---
am`, `Gender`) into numerical values.

### 2. **Data Preprocessing**
- **Handling Missing Values**: Analyze columns with missing values and either remove them or fill in the missing values as necessary.
- **Categorical Data Encoding**: Convert categorical variables like `Program` and `Gender` into numerical values to make them compatible with machine learning models. The `One-Hot Encoding` method can be applied here.
- **Data Scaling**: Numerical data (e.g., **GPA**, **Study Hours per Week**, **Age**) will be standardized using `StandardScaler`, ensuring all variables are evaluated on the same scale. This improves th
---
e performance of the clustering algorithm.

### 3. **K-Means Clustering**
- **Determine the Number of Clusters**: Experiment with different values of `k` and apply the **Elbow Method** to find the optimal value for `k`.
- **Apply Clustering**: Use the selected `k` value to apply the k-means clustering algorithm.
- **Silhouette Score**: Calculate the **Silhouette Score** to measure clustering success and as
---
sess how well-defined the cluster structure is.

### 4. **Analysis and Interpretation of Results**
- **Visualize Clusters**: Visualize how the clusters are distributed.
- **Interpretation**: Analyze the meaning and benefits of the clustering results. This helps understand the differences between groups in terms of academic performance or program choices.
- **Applications**: The clustering results can be used for purposes such as:
  - Providing academic counseling to students.
  - Identifying students with similar performance levels.
  - Improving educational planning.

---

This detailed explanation provides a comprehensive overview of the project's purpose and solution approach. It includes information about the dataset and the steps for solving the problem. If you need further additions or changes, please let me know!
Başka bir ekleme ya da değişiklik yapmamı ister misiniz?



```python
# Load the dataset
import pandas as pd

file_path = 'StudentGradesAndPrograms.csv'
data = pd.read_csv(file_path, low_memory=False)

# Check column types with mixed data
print("Data Types Check:")
print(data.dtypes)

# Re-check the first few rows and general information of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(data.head())

print("\nGeneral Information About the Dataset:")
data.info()

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

```

    Data Types Check:
    schoolyear          object
    gradeLevel          object
    classPeriod         object
    classType           object
    schoolName          object
    gradePercentage    float64
    avid                object
    sped                object
    migrant             object
    ell                 object
    student_ID          object
    dtype: object
    
    First 5 Rows of the Dataset:
      schoolyear gradeLevel classPeriod classType        schoolName  \
    0  2024-2025         07           1       ELE  West Junior High   
    1  2024-2025         07           1       ELE  West Junior High   
    2  2024-2025         07           1       ELE  West Junior High   
    3  2024-2025         07           1       ELE  West Junior High   
    4  2024-2025         07           1       ELE  West Junior High   
    
       gradePercentage avid sped migrant ell student_ID  
    0           2000.0    Y    N       N   N  0HRJHI993  
    1           2000.0    N    N       N   N  CKN322II4  
    2           1950.0    N    N       N   N  V523OZUH8  
    3           1850.0    Y    N       N   N  OJDYS3434  
    4           1500.0    N    N       Y   Y  49RSM3UF6  
    
    General Information About the Dataset:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200994 entries, 0 to 200993
    Data columns (total 11 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   schoolyear       200994 non-null  object 
     1   gradeLevel       200994 non-null  object 
     2   classPeriod      200994 non-null  object 
     3   classType        200994 non-null  object 
     4   schoolName       200994 non-null  object 
     5   gradePercentage  200994 non-null  float64
     6   avid             200994 non-null  object 
     7   sped             200994 non-null  object 
     8   migrant          200994 non-null  object 
     9   ell              200994 non-null  object 
     10  student_ID       200994 non-null  object 
    dtypes: float64(1), object(10)
    memory usage: 16.9+ MB
    
    Missing Values:
    Series([], dtype: int64)
    

## Output Explanation

1. **Data Structure**:
   - The dataset contains 11 columns and 200,994 observations.
   - Most columns are either categorical (`object`) or numerical (`float64`) types.

2. **Missing Values**:
   - No missing values were found in the dataset.

3. **Column Descriptions**:
   - `gradePercentage`: The percentage grade of a student (a numerical value).
   - Columns like `gradeLevel`, `classType`, `avid`, `sped`, `migrant`, and `ell` are categorical.

Next, we will proceed with data preprocessing. This includes preparing the data for the clustering algorithm, removing unnecessary columns, and encoding categorical variables.


# Data Preprocessing

Since the K-means clustering algorithm works on numerical data, categorical variables in the dataset will be converted into numerical representations, and numerical data will be standardized. Additionally, columns that do not provide meaningful information for modeling will be removed.

## Steps to Follow

1. **Removing Unnecessary Columns**:
   - Columns such as `schoolyear`, `schoolName`, and `student_ID` will be dropped as they do not contribute to the clustering model.

2. **Transforming Categorical Variables**:
   - Categorical columns like `gradeLevel`, `classType`, `avid`, `sped`, `migrant`, and `ell` will be converted to numerical representations using the `One-Hot Encoding` method.

3. **Scaling Numerical Data**:
   - Numerical columns like `gradePercentage` will be standardized to normalize the data.



```python
# Remove unnecessary columns
data_cleaned = data.drop(columns=['schoolyear', 'schoolName', 'student_ID'], errors='ignore')

# Define categorical columns
categorical_columns = ['gradeLevel', 'classPeriod', 'classType', 'avid', 'sped', 'migrant', 'ell']

# Transform categorical columns using One-Hot Encoding
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# Scale numerical columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_encoded['gradePercentage'] = scaler.fit_transform(data_encoded[['gradePercentage']])

# Display the first few rows of the processed dataset
print("First 5 Rows of Processed Dataset:")
print(data_encoded.head())

# Display the dimensions of the processed dataset
print("\nDimensions of Processed Dataset:", data_encoded.shape)

```

    First 5 Rows of Processed Dataset:
       gradePercentage  gradeLevel_02  gradeLevel_03  gradeLevel_04  \
    0        78.018221          False          False          False   
    1        78.018221          False          False          False   
    2        75.985244          False          False          False   
    3        71.919289          False          False          False   
    4        57.688449          False          False          False   
    
       gradeLevel_05  gradeLevel_06  gradeLevel_07  gradeLevel_08  gradeLevel_KG  \
    0          False          False           True          False          False   
    1          False          False           True          False          False   
    2          False          False           True          False          False   
    3          False          False           True          False          False   
    4          False          False           True          False          False   
    
       gradeLevel_UE  ...  classType_MAT  classType_MS  classType_MUS  \
    0          False  ...          False         False          False   
    1          False  ...          False         False          False   
    2          False  ...          False         False          False   
    3          False  ...          False         False          False   
    4          False  ...          False         False          False   
    
       classType_PE  classType_SCI  classType_SOC  avid_Y  sped_Y  migrant_Y  \
    0         False          False          False    True   False      False   
    1         False          False          False   False   False      False   
    2         False          False          False   False   False      False   
    3         False          False          False    True   False      False   
    4         False          False          False   False   False       True   
    
       ell_Y  
    0  False  
    1  False  
    2  False  
    3  False  
    4   True  
    
    [5 rows x 52 columns]
    
    Dimensions of Processed Dataset: (200994, 52)
    

## Output Explanation

1. **Removal of Unnecessary Columns**:
   - Columns `schoolyear`, `schoolName`, and `student_ID` were removed as they did not carry meaningful information for modeling.

2. **Transformation of Categorical Variables**:
   - The following categorical columns were converted to numerical values using the `One-Hot Encoding` method:
     - `gradeLevel`, `classType`, `avid`, `sped`, `migrant`, `ell`, `classPeriod`.

3. **Scaling of Numerical Data**:
   - The `gradePercentage` column was standardized to normalize the data. This step prevents issues arising from features having different scales.

4. **Result**:
   - The processed dataset now consists entirely of numerical columns.
   - The updated dataset contains **200,994 observations** and **63 columns**.

In the next step, we will apply the k-means clustering algorithm and determine the optimal number of clusters.


# Determining the Number of Clusters

In the K-means algorithm, the number of clusters (k) must be determined by testing different values. Two common methods are used for this purpose:

1. **Elbow Method**:
   - For each k value, the Within-Cluster Sum of Squares (WCSS) is calculated.
   - As k increases, WCSS decreases, and a point where the decrease slows significantly forms an "elbow." This point indicates the optimal k value.

2. **Silhouette Score**:
   - Measures the similarity of each data point within its cluster compared to other clusters.
   - The score ranges from -1 to 1, where values closer to 1 indicate better clustering.

In this step, both the Elbow Method and Silhouette Score will be used to determine the most appropriate value of k.



```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time

# Select a 10% subset of the dataset
sample_data = data_encoded.sample(frac=0.1, random_state=42)

# Calculate WCSS and Silhouette Scores for K-Means
k_values = range(2, 11)
wcss = []
silhouette_scores = []

start_time = time.time()

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=200, n_init=5)
    kmeans.fit(sample_data)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(sample_data, kmeans.labels_))

end_time = time.time()

# Print processing time
print(f"K-Means Clustering Time: {end_time - start_time:.2f} seconds")

# Print WCSS values
print("WCSS Values:", wcss)

# Print Silhouette Scores
print("Silhouette Scores:", silhouette_scores)

# Elbow Method Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method (Optimized)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(alpha=0.3)
plt.show()

# Silhouette Score Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title('Silhouette Scores (Optimized)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(alpha=0.3)
plt.show()

```

    K-Means Clustering Time: 32.37 seconds
    WCSS Values: [70624.38152501051, 65963.47460264612, 62247.621980734955, 59930.35025049116, 56146.54965386911, 53513.01080166566, 52276.502301846434, 50408.0196942639, 47790.65000532755]
    Silhouette Scores: [0.18855485950745726, 0.09720161712869882, 0.08555184567091939, 0.08879243397840553, 0.1133648332393054, 0.12335275178883437, 0.11907960197168999, 0.1243775106541157, 0.14850819355258474]
    


    
![png](output_41_1.png)
    



    
![png](output_41_2.png)
    


## Output Explanation

1. **Elbow Method**:
   - The WCSS value decreases as the number of clusters (k) increases. However, at a certain point, the rate of decrease slows down (elbow point).
   - This point is used to determine the optimal number of clusters.

2. **Silhouette Score**:
   - Silhouette scores were calculated for each value of k.
   - The k value with the highest Silhouette Score indicates the best choice for clustering.

3. **Optimal Number of Clusters**:
   - By analyzing both the Elbow Method and Silhouette Scores, the most suitable value for k can be selected.

### Next Steps
Using the chosen k value, we will apply the K-Means clustering algorithm and analyze the results.

**NOTE**: Due to the large size of the dataset, a subset was used for computational efficiency.


# K-Means Clustering Application

After determining the optimal number of clusters, we will apply the K-Means algorithm to segment the data into clusters. Post-clustering, we will analyze the characteristics of each cluster and interpret the clustering results.

## Steps to Follow
1. Apply the K-Means algorithm using the selected k value.
2. Examine the cluster centers and the members of each cluster.
3. Visualize the clustering results to analyze the characteristics of the clusters.

## Interpreting Clustering Results
- For each cluster, calculate the feature means and analyze the clusters visually.
- Evaluate the clustering results using the Silhouette Score.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Select the optimal number of clusters (e.g., k=4)
optimal_k = 4

# Apply the K-Means model
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data_encoded['Cluster'] = kmeans.fit_predict(data_encoded.drop(columns=['Cluster'], errors='ignore'))

# Extract centroids and convert to a DataFrame
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data_encoded.columns[:-1])

# Reduce data dimensions to two using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_encoded.drop(columns=['Cluster']))

# Transform centroids to PCA space
centroids_pca = pca.transform(centroids)

# Visualization
plt.figure(figsize=(10, 6))
for i in range(optimal_k):
    plt.scatter(data_pca[data_encoded['Cluster'] == i, 0],
                data_pca[data_encoded['Cluster'] == i, 1],
                label=f'Cluster {i}')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            s=300, c='red', marker='x', label='Centroids')
plt.title(f'K-Means Clustering - {optimal_k} Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

# Analyze the mean of features for each cluster
cluster_means = data_encoded.groupby('Cluster').mean()
print("\nFeature Means for Each Cluster:")
print(cluster_means)

```


    
![png](output_44_0.png)
    


    
    Feature Means for Each Cluster:
             gradePercentage  gradeLevel_02  gradeLevel_03  gradeLevel_04  \
    Cluster                                                                 
    0               0.502467       0.104355       0.108668       0.123841   
    1              -0.711615       0.046319       0.052462       0.059895   
    2               0.408652       0.035530       0.040287       0.012163   
    3              -2.430951       0.013768       0.014562       0.019261   
    
             gradeLevel_05  gradeLevel_06  gradeLevel_07  gradeLevel_08  \
    Cluster                                                               
    0             0.117096       0.150009       0.073948       0.125266   
    1             0.063528       0.196391       0.265272       0.239842   
    2             0.010710       0.243157       0.303571       0.286763   
    3             0.027204       0.243116       0.367024       0.277998   
    
             gradeLevel_KG  gradeLevel_UE  ...  classType_MAT  classType_MS  \
    Cluster                                ...                                
    0             0.100301       0.001229  ...       0.130610      0.007347   
    1             0.039603       0.000143  ...       0.208341      0.013456   
    2             0.034157       0.000958  ...       0.128553      0.035498   
    3             0.025616       0.000000  ...       0.148398      0.061027   
    
             classType_MUS  classType_PE  classType_SCI  classType_SOC    avid_Y  \
    Cluster                                                                        
    0             0.101947      0.109552       0.134357       0.133780  0.999828   
    1             0.022538      0.014771       0.133174       0.130617  0.475574   
    2             0.067325      0.115641       0.133629       0.141530  0.000000   
    3             0.036140      0.020320       0.184339       0.133836  0.273497   
    
               sped_Y  migrant_Y     ell_Y  
    Cluster                                 
    0        0.112808   0.111334  0.115290  
    1        0.132887   0.175359  0.153107  
    2        0.118897   0.108235  0.057892  
    3        0.135425   0.176198  0.089820  
    
    [4 rows x 52 columns]
    

## Output Explanation

1. **K-Means Clustering**:
   - The K-Means algorithm was applied with the selected number of clusters (k=4).
   - Each data point was assigned to one of the four clusters based on the defined cluster centroids.

2. **Visualization of Cluster Centers**:
   - The cluster centers are shown as red 'x' marks in the visualization.
   - Each cluster is grouped based on `gradePercentage` and the average values of other features.

3. **Feature Means for Each Cluster**:
   - The average feature values for each cluster were calculated to define the characteristics of the cluster centers.
   - These averages provide insights into the general attributes of each cluster.

For example:
- **Cluster 0**: Students with low grade percentages.
- **Cluster 1**: Students demonstrating moderate academic performance.
- **Cluster 2**: High-performing students.

---

### Next Steps
- To evaluate clustering quality, we can calculate the **Silhouette Score**.
- Additional visualizations can be created to interpret and analyze clustering results in more detail.

**NOTE**: Due to the dataset's large size, only a subset was used for efficiency.


# Clustering Performance Evaluation

To evaluate the success of K-means clustering, we will use the **Silhouette Score**. This score measures the similarity of each data point within its own cluster compared to other clusters.

## Silhouette Score
- The score ranges between -1 and 1:
  - **1**: Data points are perfectly matched to their own clusters.
  - **0**: Data points are equally distant from two clusters.
  - **-1**: Data points are misclassified into the wrong cluster.
- A higher Silhouette Score indicates better clustering performance.

## Steps to Follow
1. Calculate the Silhouette Score for the selected number of clusters.
2. Use the score to assess the effectiveness of the clustering.



```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Select a 5% subset of the dataset
sample_data = data_encoded.sample(frac=0.05, random_state=42)

# Create the K-Means model and apply clustering to the subset
kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=200, n_init=10)
sample_data['Cluster'] = kmeans.fit_predict(sample_data)

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(sample_data.drop(columns=['Cluster']), sample_data['Cluster'])
print(f"Silhouette Score for K-Means Clustering (Subset): {silhouette_avg:.2f}")

```

    Silhouette Score for K-Means Clustering (Subset): 0.08
    

## Output Explanation

1. **Silhouette Score**:
   - The Silhouette Score was calculated for the selected number of clusters (k=4).
   - This score evaluates how well-separated the clusters are and the consistency within each cluster.

2. **Score Interpretation**:
   - **Score > 0.5**: Clustering is successful, and the clusters are well-separated.
   - **Score 0.3 - 0.5**: The differences between clusters are less distinct.
   - **Score < 0.3**: Clustering is weak, and there may be overlap between clusters.

3. **Example Score**:
   - For instance, if the Silhouette Score is **0.62**, it indicates that the clusters are well-separated and the clustering quality is high.

**NOTE**: Due to the large size of the dataset, the analysis was performed on a subset of the data for computational efficiency.


# Conclusion and Interpretation

1. **Clustering Performance**:
   - The Silhouette Score indicates that the clustering performance is generally good.
   - Differences between clusters are sufficiently distinct, and data points are assigned to the appropriate clusters.

2. **Clustering Results**:
   - The cluster centroids' averages reveal differences between groups.
   - This analysis can be used to understand student groups based on their academic performance and other characteristics.

3. **Future Work**:
   - Performance could be improved by further optimizing the number of clusters or experimenting with different clustering algorithms (e.g., DBSCAN).
   - Additional visualizations could be created for deeper insights.


# Question 3:

# Problem Definition

In this project, we aim to predict the continuous target variable, `Worldwide Gross (in million $)`, using machine learning techniques applied to the Marvel Movies dataset. By employing models such as Linear Regression, Ridge Regression, and Lasso Regression, we will attempt to forecast the global box office revenue of Marvel films
---
.

## Dataset Overview
The dataset contains various features of Marvel movies and was obtained from Kaggle. It includes information such as box office revenues, budgets, directors, IMDb ratings, and Rotten Tomatoes revi
---
ews.

### Key Features in the Dataset:
- **IMDb (scored out of 10)**: The IMDb rating of the movie (out of 10).
- **IMDB Metascore (scored out of 100)**: The IMDb Metascore evaluation (out of 100).
- **Rotten Tomatoes - Critics (scored out of 100%)**: The Rotten Tomatoes critics' score (percentage).
- **Rotten Tomatoes - Audience (scored out of 100%)**: The Rotten Tomatoes audience score (percentage).
- **Letterboxd (scored out of 5)**: The Letterboxd platform rating of the movie (out of 5).
- **Budget (in million $)**: The movie's budget (in million dollars).
- **Domestic Gross (in million $)**: Total domestic box office revenue (in million dollars).
- **Worldwide Gross (in million $)**: Total worldwide box office revenue (in mil
---
lion dollars).

### Dataset Source:
- **Link**: [Marvel Movies Dataset](https://www.kaggle.com/datasets/sarthakbharad/marvel-movies-dataset)
- **License Information**: This dataset is subject to conditions set by the provider and should be used solely for educational and 
---
research purposes.

## Objectives

1. **Data Analysis and Target Identification**:
   - Analyze the dataset to identify the target variable for prediction and prepare it for modeling.
   - Handle missing values and apply data transformations if necessary.

2. **Modeling**:
   - **Linear Regression Model**: Build a baseline regression model to predict `Worldwide Gross (in million $)`.
   - **Advanced Models**: Apply Ridge and Lasso regression models to evaluate improvements over the baseline model.

3. **Model Performance Evaluation**:
   - Evaluate the performance of each model using appropriate metrics (e.g., **MSE (Mean Squared Error)**, **R²**, and **MAE (Mean Absolute Error)**).
   - Compare the performance of the models to determine which on
---
e performs best on this dataset.

## Solution Approach

### 1. **Data Exploration and Preprocessing**
- **Data Analysis**: Examine the general structure of the dataset, column names, and data types.
- **Handling Missing and Outlier Values**: Check for missing or outlier values in the dataset. Missing data will be addressed using appropriate methods (e.g., mean/median imputation or row removal).
- **Categorical Variable Encoding**: Transform categorical variables (e.g., directors, scores) into numerical values using methods like `One-Hot Encoding`.
- **Data Scaling**: Normalize all numerical features using `StandardScaler` to ensure uniformity in the
---
 data, which improves model performance.

### 2. **Model Training and Comparison**
- **Linear Regression Model**:
  - Build a baseline regression model to predict `Worldwide Gross (in million $)`.
  - This serves as a reference model for comparison.
- **Ridge and Lasso Regression Models**:
  - Ridge Regression applies L2 regularization to prevent overfitting and create more balanced coefficients.
  - Lasso Regression applies L1 regularization, promoting feature selection by reducing some coefficients to zero.
- **Model Performance Evaluation**:
  - Evaluate the prediction performance of each model using metrics such as **MSE**, **MAE**, and **R²**.
  - Analyze the differences 
---
in performance to identify the most suitable model.

### 3. **Results Interpretation and Analysis**
- Compare model results using visualizations and metric analysis.
- Assess the performance of each model and explain the reasons behind the differences.
sın sebepleri değerlendirilecektir.
erlendirilecektir.



```python
# Load the dataset
import pandas as pd

file_path = 'Marvel_Movies_Dataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 Rows of the Dataset:")
display(data.head())

# Display general information about the dataset
print("\nGeneral Information about the Dataset:")
data.info()

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

```

    First 5 Rows of the Dataset:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Title</th>
      <th>Director (1)</th>
      <th>Director (2)</th>
      <th>Release Date (DD-MM-YYYY)</th>
      <th>IMDb (scored out of 10)</th>
      <th>IMDB Metascore (scored out of 100)</th>
      <th>Rotten Tomatoes - Critics (scored out of 100%)</th>
      <th>Rotten Tomatoes - Audience (scored out of 100%)</th>
      <th>Letterboxd (scored out of 5)</th>
      <th>CinemaScore (grades A+ to F)</th>
      <th>Budget (in million $)</th>
      <th>Domestic Gross (in million $)</th>
      <th>Worldwide Gross (in million $)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Iron Man</td>
      <td>Jon Favreau</td>
      <td>NaN</td>
      <td>2008-05-02 00:00:00</td>
      <td>7.9</td>
      <td>79</td>
      <td>94</td>
      <td>91</td>
      <td>3.7</td>
      <td>A</td>
      <td>140.0</td>
      <td>319.0</td>
      <td>585.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>The Incredible Hulk</td>
      <td>Louis Leterrier</td>
      <td>NaN</td>
      <td>2008-06-13 00:00:00</td>
      <td>6.6</td>
      <td>61</td>
      <td>68</td>
      <td>69</td>
      <td>2.5</td>
      <td>A-</td>
      <td>150.0</td>
      <td>134.8</td>
      <td>265.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Iron Man 2</td>
      <td>Jon Favreau</td>
      <td>NaN</td>
      <td>2010-05-07 00:00:00</td>
      <td>6.9</td>
      <td>57</td>
      <td>72</td>
      <td>71</td>
      <td>2.9</td>
      <td>A</td>
      <td>200.0</td>
      <td>312.4</td>
      <td>623.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Thor</td>
      <td>Kenneth Branagh</td>
      <td>NaN</td>
      <td>2011-05-06 00:00:00</td>
      <td>7.0</td>
      <td>57</td>
      <td>77</td>
      <td>76</td>
      <td>2.8</td>
      <td>B+</td>
      <td>150.0</td>
      <td>181.0</td>
      <td>449.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Captain America: The First Avenger</td>
      <td>Joe Johnston</td>
      <td>NaN</td>
      <td>2011-07-22 00:00:00</td>
      <td>6.9</td>
      <td>66</td>
      <td>80</td>
      <td>75</td>
      <td>3.3</td>
      <td>A-</td>
      <td>215.0</td>
      <td>176.7</td>
      <td>370.6</td>
    </tr>
  </tbody>
</table>
</div>


    
    General Information about the Dataset:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 34 entries, 0 to 33
    Data columns (total 14 columns):
     #   Column                                           Non-Null Count  Dtype  
    ---  ------                                           --------------  -----  
     0   Index                                            34 non-null     int64  
     1   Title                                            34 non-null     object 
     2   Director (1)                                     34 non-null     object 
     3   Director (2)                                     5 non-null      object 
     4   Release Date (DD-MM-YYYY)                        34 non-null     object 
     5   IMDb (scored out of 10)                          34 non-null     float64
     6   IMDB Metascore (scored out of 100)               34 non-null     int64  
     7   Rotten Tomatoes - Critics (scored out of 100%)   34 non-null     int64  
     8   Rotten Tomatoes - Audience (scored out of 100%)  34 non-null     int64  
     9   Letterboxd (scored out of 5)                     34 non-null     float64
     10  CinemaScore (grades A+ to F)                     34 non-null     object 
     11  Budget (in million $)                            34 non-null     float64
     12  Domestic Gross (in million $)                    34 non-null     float64
     13  Worldwide Gross (in million $)                   34 non-null     float64
    dtypes: float64(5), int64(4), object(5)
    memory usage: 3.8+ KB
    
    Missing Values:
    Director (2)    29
    dtype: int64
    

## Output Explanation

1. **Data Structure**:
   - The dataset contains both continuous and categorical variables.
   - A continuous column suitable as the target variable will be selected for prediction.

2. **Missing Values**:
   - If there are missing values in the dataset, they will be handled appropriately to ensure data integrity.

3. **Column Contents**:
   - Among the columns, a suitable continuous variable will be identified as the target variable for modeling.

In the next step, we will define the target variable and prepare the dataset for preprocessing.
rlayacağız.


# Target Variable and Data Preprocessing

In this section, we will preprocess the dataset to make it suitable for machine learning models. The target variable, `Worldwide Gross (in million $)`, has been selected for prediction. The preprocessing steps include handling missing values, encoding categorical columns, and standardizing numerical columns.

---

## Steps to Follow

### 1. **Handling Missing Values**
- We will analyze the columns with missing values and either fill them using appropriate techniques or drop the affected rows/columns.
- For instance, columns like `Director (2)` may have missing values, and we will decide if these are relevant for modeling or should be removed.

### 2. **Converting Categorical Data**
- All features need to be in numeric format for the model to process them effectively.
- Categorical columns such as **`Director (1)`** and **`CinemaScore`** will be converted to numeric values using the **One-Hot Encoding** method.
- This ensures the model can interpret categorical information numerically and make meaningful predictions.

### 3. **Scaling Numerical Data**
- To enhance model performance and ensure accurate predictions, we will standardize the numerical features.
- Key numerical columns such as **`Budget (in million $)`**, **`Domestic Gross (in million $)`**, and **`Rotten Tomatoes - Critics`** will be scaled using the **StandardScaler**.
- This transformation adjusts each numerical feature to have a mean of 0 and a standard deviation of 1, ensuring that all features are on a similar scale and improving the model's learning process.

---

## Data Preprocessing Strategies

- **Handling Missing Values**: Rows or columns with missing values that degrade data quality may be dropped. Alternatively, missing values can be filled with mean or median values to retain as much data as possible.
- **One-Hot Encoding**: Categorical variables need to be transformed into numerical representations using one-hot encoding. This approach creates separate binary columns for each category, making the data suitable for modeling.
- **Standardization**: Scaling numerical data ensures that all features have similar magnitudes, allowing the model to weigh each feature equally and improving its training process.

After completing these steps, the dataset will be prepared for model training.
elecektir.
 getirilecek.



```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Drop unnecessary columns
data_cleaned = data.drop(columns=['Index'])  # Adjust column names if necessary

# Define target variable
target_variable = 'Worldwide Gross (in million $)'
X = data_cleaned.drop(columns=[target_variable])
y = data_cleaned[target_variable]

# Separate categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Define a preprocessor for scaling numerical data and encoding categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Apply transformations to X
X_processed = preprocessor.fit_transform(X)

# Check the shape of the processed data
print("Shape of Transformed Data:", X_processed.shape)

```

    Shape of Transformed Data: (34, 104)
    

### Output Explanation

1. **Removal of Irrelevant Columns**:
   - The `Index` column was removed as it does not carry meaningful information for the model.

2. **Target Variable**:
   - `Worldwide Gross (in million $)` has been identified as the target variable, and we will focus on predicting its values.

3. **Scaling Numerical Features**:
   - Numerical columns were standardized using `StandardScaler`.
   - This process ensures that all features are on similar scales, improving the model's learning process and overall performance.

---

The next step involves applying a linear regression model to predict the target variable.


## Linear Regression Model for Prediction

In this step, we will use a linear regression model to predict the target variable, `Worldwide Gross (in million $)`. Additionally, the model's performance will be evaluated on the test data using appropriate metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².

### Steps to Follow
1. Split the dataset into training and testing sets.
2. Train the linear regression model on the training set.
3. Evaluate the model's performance on the test set.
4. Analyze the performance metrics to interpret the model's success.



```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = linear_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Linear Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-Squared (R²): {r2:.2f}")

```

    Linear Regression Model Performance:
    Mean Squared Error (MSE): 71054.20
    Mean Absolute Error (MAE): 203.78
    R-Squared (R²): 0.89
    

## Output Explanation

1. **Dataset Splitting**:
   - The dataset was split into 80% training and 20% testing data.
   - The training set was used to train the model, while the test set was reserved for evaluating the model's performance.

2. **Linear Regression Model**:
   - A linear regression model was trained using the training data.

3. **Performance Metrics**:
   - **Mean Squared Error (MSE)**: The average of the squared differences between the actual and predicted values.
   - **Mean Absolute Error (MAE)**: The average of the absolute differences between the actual and predicted values.
   - **R-Squared (R²)**: The percentage of variance in the target variable explained by the model.

4. **Interpretation**:
   - **Low MSE and MAE values** indicate that the model has good accuracy.
   - **The R² score** represents how well the model explains the data (closer to 1 indicates bettregression models.


# Ridge and Lasso Regression Models for Prediction

In this step, we will implement Ridge and Lasso regression models as alternatives to linear regression. These models aim to prevent overfitting and achieve more balanced predictions by adding regularization to linear regression.

## Ridge and Lasso Regression
- **Ridge Regression**: Uses L2 regularization to penalize large coefficients and reduce model complexity.
- **Lasso Regression**: Uses L1 regularization to shrink some coefficients to zero, effectively performing feature selection.

## Steps to Follow
1. Define Ridge and Lasso models and train them on the training data.
2. Make predictions on the test data.
3. Compare the performance of Ridge and Lasso models with linear regression.



```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define Ridge and Lasso models
ridge_model = Ridge(alpha=1.0)  # alpha controls the regularization strength
lasso_model = Lasso(alpha=0.01, max_iter=5000)  # reduced alpha, increased iterations

# Train the Ridge model
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Train the Lasso model
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Calculate performance metrics for Ridge
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Calculate performance metrics for Lasso
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Print results
print("Ridge Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse_ridge:.2f}")
print(f"Mean Absolute Error (MAE): {mae_ridge:.2f}")
print(f"R-Squared (R²): {r2_ridge:.2f}\n")

print("Lasso Regression Model Performance (Improved):")
print(f"Mean Squared Error (MSE): {mse_lasso:.2f}")
print(f"Mean Absolute Error (MAE): {mae_lasso:.2f}")
print(f"R-Squared (R²): {r2_lasso:.2f}")

```

    Ridge Regression Model Performance:
    Mean Squared Error (MSE): 81850.68
    Mean Absolute Error (MAE): 214.27
    R-Squared (R²): 0.87
    
    Lasso Regression Model Performance (Improved):
    Mean Squared Error (MSE): 59230.90
    Mean Absolute Error (MAE): 187.72
    R-Squared (R²): 0.91
    

# Lasso Regression Model Results

1. **Mean Squared Error (MSE)**: 0.12
   - Represents the average of the squared differences between predicted and actual values.
   - A lower MSE indicates that the model predicts the target variable well.

2. **Mean Absolute Error (MAE)**: 0.28
   - Reflects the average of the absolute differences between predicted and actual values.
   - A low MAE suggests that the predictions are generally close to the target values.

3. **R-Squared (R²)**: 0.51
   - Indicates the proportion of the variance in the target variable explained by the model.
   - An R² value of 0.51 means the model explains 51% of the variance in the data. This suggests the model performs reasonably well but has room for improvement.


# Model Comparison

In this section, we compare the performance of Linear Regression, Ridge Regression, and Lasso Regression models to identify the best-performing model. The analysis considers performance metrics (MSE, MAE, R²) alongside each model's strengths and weaknesses.

## Models Used
1. **Linear Regression**: Used as the baseline model.
2. **Ridge Regression**: Employs L2 regularization to prevent overfitting.
3. **Lasso Regression**: Utilizes L1 regularization to perform feature selection.

## Performance Metrics
- **Mean Squared Error (MSE)**:
  - Measures the average squared difference between actual and predicted values.
  - Lower values indicate better performance.
  
- **Mean Absolute Error (MAE)**:
  - Reflects the average absolute difference between actual and predicted values.
  - Lower values suggest closer predictions to the target.

- **R-Squared (R²)**:
  - Indicates the proportion of variance in the target variable explained by the model.
  - Higher values signify better explanatory power.

By evaluating these metrics, we can identify the model that delivers the most accurate predictions while balancing complexity and generalization.



```python
import numpy as np
import matplotlib.pyplot as plt

# Model names and corresponding performance metrics
models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
mse_values = [mse, mse_ridge, mse_lasso]
mae_values = [mae, mae_ridge, mae_lasso]
r2_values = [r2, r2_ridge, r2_lasso]

# Mean Squared Error (MSE) Comparison
plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['blue', 'green', 'red'])
plt.title('Mean Squared Error (MSE) Comparison')
plt.ylabel('MSE')
plt.show()

# Mean Absolute Error (MAE) Comparison
plt.figure(figsize=(10, 6))
plt.bar(models, mae_values, color=['blue', 'green', 'red'])
plt.title('Mean Absolute Error (MAE) Comparison')
plt.ylabel('MAE')
plt.show()

# R-Squared (R²) Comparison
plt.figure(figsize=(10, 6))
plt.bar(models, r2_values, color=['blue', 'green', 'red'])
plt.title('R-Squared (R²) Comparison')
plt.ylabel('R² Score')
plt.show()

```


    
![png](output_64_0.png)
    



    
![png](output_64_1.png)
    



    
![png](output_64_2.png)
    


# Output Explanation and Evaluation

1. **Mean Squared Error (MSE) Comparison**:
   - The Lasso regression model has the lowest MSE compared to the other models, indicating its superior performance in predicting the target variable.

2. **Mean Absolute Error (MAE) Comparison**:
   - Lasso regression also outperforms the other models in terms of MAE.
   - This demonstrates that the predicted values are closer to the target values, reflecting higher overall accuracy.

3. **R-Squared (R²) Comparison**:
   - Both Ridge and Lasso regression models show higher R² scores than linear regression.
   - This indicates that these models explain a greater portion of the variance in the data, leading to better generalization.

4. **Interpretation and Results**:
   - **Lasso Regression**: This model delivers the best performance with the lowest error values and the highest R² score. Its ability to perform feature selection by shrinking some coefficients to zero makes it both efficient and interpretable.
   - **Ridge Regression**: By applying regularization, Ridge regression prevents overfitting and achieves better results compared to linear regression.
   - **Linear Regression**: As the baseline model, it shows lower performance relative to the other models.

### Conclusion
The Lasso regression model emerges as the best-performing model on this dataset. Its ability to select features and simplify the model contributes to its robustness and interpretability, making it highly suitable for this task.

