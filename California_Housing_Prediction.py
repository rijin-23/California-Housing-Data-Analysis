#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries and Dataset

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[76]:


housing = pd.read_csv("housing.csv")


np.random.seed(42)

#Shuffle the indices of the dataset using np.random.permutation.
#Determine the training size
#Fetch training indices and testing indices
#Return the records using iloc and not the indices
def train_test_split(data, test_size_ratio):
    shuffleindices = np.random.permutation(len(data))
    train_size = int(len(data)*(1-test_size_ratio))
    train_indices = shuffleindices[:train_size]
    print(train_indices)
    test_indices = shuffleindices[train_size:]
    print(test_indices)
    return data.iloc[train_indices], data.iloc[test_indices]



train,test = train_test_split(housing,0.2)


housing["median_income_cat"] = pd.cut(housing["median_income"], bins=[0.,1.5,3.,4.5,6.,np.inf], labels=[1,2,3,4,5])


# In[86]:


#housing["median_income_cat"].hist()


# Performing Stratified Sampling so that the sample data is representative

# In[87]:


from sklearn.model_selection import StratifiedShuffleSplit

#We want to split the hosing dataset into 1 set of train and test set, hence n_split = 1. If n_split = 2, the split method will
#create 2 sets of train and test with a proportion of 0.8 in train and 0.2 test in both the sets.
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

#Fetching the indices for the training and testing set. split.split is a method for the above variable split and it demands the
#dataset to be split(housing) and what column to refer for stratified sampling(housing["median_income_cat"])
for train_index, test_index in split.split(housing, housing["median_income_cat"]):
    strat_train = housing.loc[train_index]
    strat_test = housing.loc[test_index]


# In[88]:


#To see the proportion of each income category in the dataset
#housing["median_income_cat"].value_counts() / len(housing)


# In[89]:


#strat_train.head()


# In[90]:


#strat_train["median_income_cat"].value_counts() / len(strat_train)


# In[91]:


#strat_test["median_income_cat"].value_counts() / len(strat_test)


# Looking at the proprtion of the housing dataset and the stratified training and test sets it is evident that proportions are matching. This is what we were trying to achieve.

# In[92]:


#Removing the median_income_cat feature to make the dataset the same as before.
for x in (strat_train, strat_test):
    x.drop("median_income_cat", axis=1, inplace=True)


# In[93]:


#strat_train.head()


# In[94]:


housing = strat_train.copy()


# ### Data Visualization

# In[95]:


#plt.scatter(housing["longitude"], housing["latitude"])


# As seen above, it is difficult to see a pattern here.

# In[96]:


#plt.scatter(housing["longitude"], housing["latitude"], alpha = 0.1)


# Setting alpha to 0.1 helps us visualize data in a better way. We can see that the most of the districts are from bay area and a good number of districts in central part of california.
# 
# Now lets visualize data on the basis of the population size and our label "median_house_value".

# In[97]:


'''sc = plt.scatter(housing["longitude"], housing["latitude"], s = housing["population"]/100, c = housing["median_house_value"], label = "population", cmap= plt.get_cmap("jet"), alpha = 0.3)
plt.colorbar(sc)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()'''


# It is visible that the houses in the bay areas are much more costlier(dots in red) compared to the houses elsewhere. But, in the northern bay areas, the median house prices tend to have a similar value to that of the houses in the central California.

# In[98]:


#Correlation among features
#correlation = housing.loc[:,housing.columns!="ocean_proximity"].corr()


# In[99]:


#correlation["median_house_value"].sort_values(ascending=False)


# As the Pearson's correlation coefficeint goes towards 1, there is a positive relationship among the feature(s) and the label(s) and vice versa. If a correlation coeff is closer to 0 that indicates no correlation

# In[100]:


'''import seaborn as sns

sns.heatmap(correlation)'''


# In[101]:


#Scatter matrix for correlation between every features. We'll use the positively correlated features
'''from pandas.plotting import scatter_matrix

scatter_matrix(housing[["median_house_value","median_income", "total_rooms", "housing_median_age"]], figsize=(20,10))

'''
# In[102]:


#Scatter matrix with seaborn
#sns.pairplot(housing[["median_house_value","median_income", "total_rooms", "housing_median_age"]])


# Median income is highly correlated with the target label. 

# In[103]:


#plt.scatter(housing["median_income"], housing["median_house_value"], s = 5, alpha=0.5)


# #### Experimenting with attribute combinations
# Features such as total rooms, total bedrooms can be of much more importance if we can get to know the total numer of bedrooms per rooms or total rooms per household.

# In[104]:


housing["bedrooms_per_rooms"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["population_per_household"] = housing["population"] / housing["households"]


# In[105]:


#correlation = housing.loc[:,housing.columns!="ocean_proximity"].corr()


# In[106]:


#correlation["median_house_value"].sort_values(ascending=False)


# The calculated attribute "rooms_per_household" shows a positive correlation with the target label. It means, more the numbers of rooms then more the median house value. In contrast, bedrooms_per_rooms show a negative correlation to the target label. Less the no. bedrooms, more the price of the house.

# #### Splitting the features and the label

# In[107]:


#housing


# In[108]:


housing = strat_train.drop("median_house_value", axis=1)
housing_target = strat_train["median_house_value"].copy()


# ### Data Cleaning

# In[109]:


#housing.isna().sum()


# In[110]:


#plt.hist(housing["total_bedrooms"], bins=20)


# As the distribution is right skewed, it is wise to fill the na values with the attribute median.
# 
# Storing the median of the total bedrooms so that it can be utilized in the test set too.

# In[111]:


total_bedrooms_median = housing["total_bedrooms"].median()
housing.fillna(total_bedrooms_median, inplace=True)


# In[112]:


#housing.isna().sum()


# Alternatively, scikits learn's SimpleImputer class can also be used to fill the null values

# In[113]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_no_cat = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_no_cat)
imputer.statistics_


# It is better to use simple imputer and calculate all the median values as it may be possible that the null values may exist in the production data and hence it can be transformed with the imputer instance we trained.

# In[114]:


X = imputer.transform(housing_no_cat)


# In[115]:


housing_tr = pd.DataFrame(X, columns=housing_no_cat.columns)


# In[116]:


#housing_tr


# #### Handling Categorical values

# There is a need to convert the categorical values to numbers as most of the ML algorithms work well with numbers and not with text data.

# In[117]:


from sklearn.preprocessing import OneHotEncoder
housing_cat = housing[["ocean_proximity"]]
ohencoder = OneHotEncoder()
new_housing_cat = ohencoder.fit_transform(housing_cat)
new_housing_cat


# In[118]:


new_housing_cat.toarray()


# #### Transformation Pipeline

# In[119]:


from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[120]:


col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names]


# In[121]:


housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# In[122]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline= Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_no_cat)


# In[123]:


#housing_no_cat


# In[124]:


from sklearn.compose import ColumnTransformer

num_attribute = list(housing_no_cat)
cat_attribute = ["ocean_proximity"]
num_attribute
full_pipeline = ColumnTransformer([
    ('num_tran', num_pipeline, num_attribute),
    ('cat_tran', OneHotEncoder(), cat_attribute)
]) 

housing_prepared = full_pipeline.fit_transform(housing)


# In[125]:


#pd.DataFrame(housing_prepared)


# ### Training the model

# In[126]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(housing_prepared, housing_target)

def data_prep_and_pred(data):
    data = pd.DataFrame(data, columns=housing.columns)
    cleaned = full_pipeline.transform(data)
    return model.predict(cleaned)
    
# In[137]:


'''from sklearn.metrics import mean_squared_error

predictions = model.predict(housing_prepared)
mse = mean_squared_error(housing_target, predictions)
rmse = np.sqrt(mse)
rmse


# The prediction error is of $68627. This may be due to underfitting. To solve this wither we need more data, or remove noises or try a more complex model that can find non linear relationships in the data. Log transformations can also be applied to achieve better results.

# In[138]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(housing_prepared, housing_target)


# In[139]:


dt_predictions = dt.predict(housing_prepared)
dt_mse = mean_squared_error(housing_target, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_rmse


# 0 error! Means there is some sort of overfitting. It is not advisable to run the test set yet as it should be the final step.

# ### Cross Validation

# In[140]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt, housing_prepared, housing_target, scoring="neg_mean_squared_error", cv=10)

dt_rmse_scores = np.sqrt(-scores)


# In[141]:


print(dt_rmse_scores)
print(dt_rmse_scores.mean())
print(dt_rmse_scores.std())


# The mean RMSE score for the cross validation is around $71432 which is worse than linear regression. This is an example of the model overfitting on the training set.
# 
# Let's try cross validation for Linear Regression

# In[ ]:


lin_scores = cross_val_score(model, housing_prepared, housing_target, scoring="neg_mean_squared_error", cv=10)


# In[147]:


lin_scores_rmse = np.sqrt(-lin_scores)


# In[148]:


def display_scores(score):
    print("Scores: ",score)
    print("Mean:",score.mean())
    print("Standard Deviation",score.std())
    
display_scores(lin_scores_rmse)


# It is almost the same as before, but the error is better than decision trees.

# In[149]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(housing_prepared, housing_target)


# In[150]:


rf_prediction = rf.predict(housing_prepared)
rf_mse = mean_squared_error(housing_target, rf_prediction)
rf_rmse = np.sqrt(rf_mse)
rf_rmse


# Looks much better than both the previously used algorithms. Let's cross validate

# In[154]:


rf_scores = cross_val_score(rf, housing_prepared, housing_target, scoring="neg_mean_squared_error", cv=10)


# In[155]:


rf_rmse_score = np.sqrt(-rf_scores)


# In[156]:


display_scores(rf_rmse_score)


# The mean RMSE is $50435

# In[ ]:'''




