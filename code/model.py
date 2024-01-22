#!/usr/bin/env python
# coding: utf-8

# In[14]:


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


# In[15]:


# import data
df = pd.read_csv('data/processed/merchant_data.csv')
print(df.groupby('label')['merchant'].count()) 

# fill active days nan with 0
df['active_days'] = df['active_days'].fillna(0)
# fill average days beteen charges nan with  65
df['avg_days_between'] = df['avg_days_between'].fillna(65)

# Define your feature matrix X and target vector y
X = df.drop(columns=['merchant', 'label', 'first_charge_date', 'first_payment_date'])  # drop non-feature columns
# drop columns whose column name contains '_mean','_std', '_cv'

y = df['label']
all_variable = X.columns.tolist()
categorical_vars = ['industry', 'country', 'business_size']
continuous_vars = [i for i in all_variable if i not in categorical_vars]


# In[16]:


# data preprocessing and normalization
# Imputer for continuous variables
continuous_imputer = SimpleImputer(strategy='constant', fill_value=0)  
# Imputer for categorical variables
categorical_imputer = SimpleImputer(strategy='constant', fill_value='Missing')  # or 'most_frequent'

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', continuous_imputer),  # First impute missing values
            ('scaler', StandardScaler())  # Then scale
        ]), continuous_vars),
        
        ('cat', Pipeline([
            ('imputer', categorical_imputer),  # First impute missing values
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Then encode
        ]), categorical_vars)
    ])


# In[17]:


# Split data into training and testing sets
# it is highly imbalance, so we need to stratify the split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


# Fit the preprocessor on the training data
preprocessor.fit(X_train)
# Transform both training and test data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# In[19]:


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
# }
space = {
    'max_depth': hp.quniform('max_depth', 4, 10, 1),  # Adjusted for potential underfitting
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),  # Lower end for more models
    'num_boost_round': hp.quniform('num_boost_round', 100, 600, 50),  # More rounds
    'gamma': hp.uniform('gamma', 0.0, 0.5),  # Fine as is
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),  # Fine as is
    'subsample': hp.uniform('subsample', 0.6, 1),  # Adjusted range
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),  # Adjusted range
    'scale_pos_weight': scale_pos_weight  # Fine as is
}


# In[20]:


def objective(params):
    params_list = {
        'max_depth': int(params['max_depth']),
        'learning_rate': params['learning_rate'],
        'gamma': params['gamma'],
        'min_child_weight': int(params['min_child_weight']),
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'objective': 'binary:logistic',
        'scale_pos_weight': params['scale_pos_weight']
    }
    num_boost_round = int(params['num_boost_round'])
    # Initialize an empty list to hold AUC scores
    val_auc_scores = []
    train_auc_scores = []
    
    # Create a StratifiedKFold object
    kf = StratifiedKFold(n_splits=5)
    
    for train_index, val_index in kf.split(X_train_transformed, y_train):
        X_train_k, X_val_k = X_train_transformed[train_index], X_train_transformed[val_index]
        y_train_k, y_val_k = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Convert the dataset into an optimized data structure called Dmatrix that XGBoost supports
        dtrain = xgb.DMatrix(X_train_k, label=y_train_k)
        dval = xgb.DMatrix(X_val_k, label=y_val_k)

        # Train the model with early stopping
        evals = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(params_list, dtrain, evals=evals, num_boost_round = num_boost_round, early_stopping_rounds=100, verbose_eval=False)
        
        # Predict on the validation set using the best iteration
        best_iteration = model.best_iteration
        preds_val_k = model.predict(dval, iteration_range=(0, best_iteration))
        preds_train_k = model.predict(dtrain, iteration_range=(0, best_iteration))
        
        # Calculate and append the performance metric on the validation set
        auc_val_k = roc_auc_score(y_val_k, preds_val_k)
        val_auc_scores.append(auc_val_k)
        auc_train_k = roc_auc_score(y_train_k, preds_train_k)
        train_auc_scores.append(auc_train_k)
    # Calculate the average AUC score across all folds
    avg_auc = np.mean(val_auc_scores)
    avg_auc_train = np.mean(train_auc_scores)
    #print('Train AUC: {:.5f}, Val AUC: {:.5f}'.format(avg_auc_train, avg_auc))
    return {'loss': -avg_auc, 'status': STATUS_OK}


# In[21]:


# Run hyperparameter tuning
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)


# In[35]:


# Convert hyperparameters to appropriate types
best['max_depth'] = int(best['max_depth'])
best['min_child_weight'] = int(best['min_child_weight'])
best['num_boost_round'] = int(best['num_boost_round'])
best['objective'] = 'binary:logistic'
best['scale_pos_weight'] = scale_pos_weight
best['eval_metric'] = 'auc'


# In[47]:


# Prepare data for XGBoost format
dtrain_full = xgb.DMatrix(X_train_transformed, label=y_train)
dtest = xgb.DMatrix(X_test_transformed)
# Train the final model on the full training data
# num_boost_round = best.pop('num_boost_round')
final_model = xgb.train(best, dtrain_full, num_boost_round=best['num_boost_round'])
# save model
final_model.save_model('model/xgb.model')
# Predict on the test set
final_preds = final_model.predict(dtest)

# Calculate the final performance metric on the test set
final_auc = roc_auc_score(y_test, final_preds)
print(f"The final ROC AUC on the test set is: {final_auc}")

# based on the best F1 score find the best threshold
thresholds = np.arange(0.1, 1, 0.02)
f1_scores = []
for thresh in thresholds:
    y_pred_binary = [1 if prob > thresh else 0 for prob in final_preds]
    f1_scores.append(f1_score(y_test, y_pred_binary))

print('f1 score:' + str(np.max(f1_scores)))
threshold = thresholds[np.argmax(f1_scores)]
print('threshold:' + str(threshold))

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
y_pred_binary = [1 if prob > threshold else 0 for prob in final_preds]

# Calculate Recall
test_recall = recall_score(y_test, y_pred_binary)
print(f"Recall on test set: {test_recall}")

# Calculate Precision
test_precision = precision_score(y_test, y_pred_binary)
print(f"Precision on test set: {test_precision}")

# Calculate Accuracy
test_accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy on test set: {test_accuracy}")

# Calculate F1 score
test_f1 = f1_score(y_test, y_pred_binary)
print(f"F1 score on test set: {test_f1}")


# In[44]:


# Propensity to use Subscription on the data with label 0
X_no_subscription = X[y == 0]
print(X_no_subscription.shape)
X_no_subscription_transformed = preprocessor.transform(X_no_subscription)
d_no_subscription = xgb.DMatrix(X_no_subscription_transformed)
preds_no_subscription = final_model.predict(d_no_subscription)

# combine with merchant name 
merchant_name = df[y == 0]['merchant']
merchant_name = merchant_name.reset_index(drop=True)
preds_no_subscription = pd.DataFrame(preds_no_subscription, columns=['propensity'])
preds_no_subscription = pd.concat([merchant_name, preds_no_subscription], axis=1)
preds_no_subscription = preds_no_subscription.sort_values(by=['propensity'], ascending=False)
preds_no_subscription.to_csv('data/result/merchant_propensity_subscription.csv', index=False)

# select the merchant with propensity > threshold
merchant_selected = preds_no_subscription[preds_no_subscription['propensity'] > threshold]
print(merchant_selected.shape)
merchant_selected.to_csv('data/result/merchant_selected.csv', index=False)

