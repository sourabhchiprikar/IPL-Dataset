import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


# loading dataset
ipl_auction_df = pd.read_csv("IPL IMB381IPL2013.csv")

pd.set_option('display.max_columns',7) # number of coloumns to display

print("display 5 rows:")

ipl_auction_df.head(5)

ipl_auction_df.info()

ipl_auction_df.iloc[0:5, 0:10]

ipl_auction_df.iloc[0:5, 13:]

X_features = ipl_auction_df.columns

X_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'T-RUNS', 'T-WKTS', 'ODI-RUNS-S', 'ODI-SR-B', 'ODI-WKTS', 'ODI-SR-BL', 'CAPTAINCY EXP', 'RUNS-S', 'HS', 'AVE', 'SR-B', 'SIXERS', 'RUNS-C', 'WKTS', 'AVE-BL', 'ECON', 'SR-BL']

# Encoding Categorical Features
ipl_auction_df['PLAYING ROLE'].unique()
"""array(['Allrounder', 'Bowler', 'Batsman', 'W. Keeper'], dtype=object)"""

pd.get_dummies(ipl_auction_df['PLAYING ROLE'])[0:5]

categorical_features = ['AGE', 'COUNTRY', 'PLAYING ROLE', 'CAPTAINCY EXP']

ipl_auction_encoded_df = pd.get_dummies( ipl_auction_df[X_features],columns = categorical_features,drop_first = True )

ipl_auction_encoded_df.columns
X_features = ipl_auction_encoded_df.columns

# Add column of all ones and train test split
import statsmodels.api as sm
#import numpy as np

X = sm.add_constant( ipl_auction_encoded_df )
Y = ipl_auction_df['SOLD PRICE']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split( X ,Y,train_size = 0.8,random_state = 42 )

# Fitting the Model
ipl_model_1 = sm.OLS(train_y, train_X).fit()
ipl_model_1.summary2()
# http://efavdb.com/interpret-linear-regression/

# HS, AGE_2, AVE and COUNTRY_ENG is significant
# not very intuitive and can be result of multi-collinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Method to calculate Variance inflation factor 
def get_vif_factors( X ):
    X_matrix = X.as_matrix()
    vif = [ variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]
    vif_factors = pd.DataFrame()
    vif_factors['column'] = X.columns
    vif_factors['vif'] = vif
    return vif_factors

vif_factors = get_vif_factors( X[X_features] )
vif_factors

# Select the features that have VIF value more than 4
columns_with_large_vif = vif_factors[vif_factors.vif > 4].column

# Plot the heatmap for features with moore than 4
plt.figure( figsize = (12,10) )
sn.heatmap( X[columns_with_large_vif].corr(), annot = True );
plt.title( "Figure 4.5 - Heatmap depicting correlation between features");

# T-RUNS and ODI-RUN-S, ODI-WKTS and T-WKTS - high correlation
# RUNS-S, HS, AVE, SIXERS - high correlation
# AVE-BL, ECON and SR-BL - high correlation

# Remove some features
columns_to_be_removed = ['T-RUNS', 'T-WKTS', 'RUNS-S', 'HS', 'AVE', 'RUNS-C', 'SR-B', 'AVE-BL', 'ECON', 'ODI-SR-B', 'ODI-RUNS-S', 'AGE_2', 'SR-BL']
X_new_features = list( set(X_features) - set(columns_to_be_removed) )

# get VIF for new set of features
get_vif_factors( X[X_new_features] )
# all VIFs are less than 4 - no more multi-collinearity

# Building a new model after removing multicollinearity
train_X = train_X[['const']+X_new_features]
ipl_model_2 = sm.OLS(train_y, train_X).fit()
ipl_model_2.summary2()

significant_vars = ['COUNTRY_IND', 'COUNTRY_ENG', 'SIXERS', 'CAPTAINCY EXP_1']
train_X = train_X[significant_vars]
ipl_model_3 = sm.OLS(train_y, train_X).fit()
ipl_model_3.summary2()

### Residual Analysis

# P-P Plot
def draw_pp_plot( model, title ):
    probplot = sm.ProbPlot( model.resid );
    plt.figure( figsize = (8, 6) );
    probplot.ppplot( line='45' );
    plt.title( title );
    plt.show();
    
draw_pp_plot( ipl_model_3,"Figure 4.6 - Normal P-P Plot of Regression Standardized Residuals");

# Homoscedasticity
def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()

def plot_resid_fitted( fitted, resid, title):
    plt.scatter( get_standardized_values( fitted ),get_standardized_values( resid ) )
    plt.title( title )
    plt.xlabel( "Standardized predicted values")
    plt.ylabel( "Standardized residual values")
    plt.show()

plot_resid_fitted( ipl_model_3.fittedvalues,ipl_model_3.resid,"Figure 4.7 - Residual Plot")
#Because the third dummy can be explained as the linear combination of the first two: FL = 1 - (CA + NY)

# leverage plot - high influencial data points
# Leverage value of more than 3(k+1)/n are treated as highly influential observation
k = train_X.shape[1]
n = train_X.shape[0]

print( "Number of variables:", k, " and number of observations:", n)

leverage_cutoff = 3*((k + 1)/n)
print( "Cutoff for leverage value: ", round(leverage_cutoff, 3) )

from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(8,6) )
influence_plot( ipl_model_3, ax = ax )
plt.title( "Figure 4.7 - Leverage Value Vs Residuals")
plt.show()

# 23, 58, 83 have comparitively high leverage with residuals
# we can filter out influencial observations
ipl_auction_df[ipl_auction_df.index.isin( [23, 58, 83] )]

# These observations do not have high residuals. So, may not be necessary to remove.
# if the rows has to be removed:
train_X_new = train_X.drop( [23, 58, 83], axis = 0)
train_y_new = train_y.drop( [23, 58, 83], axis = 0)

# building the model again
"""ipl_model_f = sm.OLS(train_y_new, train_X_new).fit()
ipl_model_f.summary2()"""

# Cook's Distance
ipl_influence = ipl_model_3.get_influence()
(c, p) = ipl_influence.cooks_distance
plt.stem( np.arange( len( train_X) ),np.round( c, 3 ),markerfmt="," );
plt.title( "Figure 4.3 - Cooks distance for all observations in IPL auction dataset" );
plt.xlabel( "Row index")
plt.ylabel( "Cooks Distance");

# Z-Score
from scipy.stats import zscore
ipl_auction_df['z_score_soldprice'] = zscore( ipl_auction_df['SOLD PRICE'] )
ipl_auction_df[ (ipl_auction_df['z_score_soldprice'] > 3.0) | (ipl_auction_df['z_score_soldprice'] < -3.0) ]

### Making predictions on validation set
pred_y = ipl_model_3.predict( test_X[train_X.columns] )

#Measuring RMSE
from sklearn import metrics
metrics.mean_squared_error(pred_y, test_y)

#Measuring R-squared value
metrics.r2_score(pred_y, test_y)


# Transforming Response Variable
train_y = np.sqrt( train_y )
ipl_model_4 = sm.OLS(train_y, train_X).fit()
ipl_model_4.summary2()

"""
The r-squard value of the model has increased to 0.751. And the following P-P plot also shows that the
residuals follow a normal distribution.
"""

draw_pp_plot( ipl_model_4,"Figure 4.8 - Normal P-P Plot of Regression Standardized Residuals");

### Making predictions on validation set
pred_y = np.power( ipl_model_4.predict( test_X[train_X.columns] ), 2)

#Measuring RMSE
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(pred_y, test_y))

#Measuring R-squared value
np.round( metrics.r2_score(pred_y, test_y), 2 )

### Using sklearn ######
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lin_model = LinearRegression()
lin_model.fit(train_X, train_y)

print("Parameters of the model:::")
print("Coeficient:", lin_model.coef_)
print("Intercept:",lin_model.intercept_)

# model evaluation for training set
y_train_predict = lin_model.predict(train_X)
rmse = (np.sqrt(mean_squared_error(train_y, y_train_predict)))
r2 = r2_score(train_y, y_train_predict)
r2_1 = lin_model.score(train_X, train_y) # 2nd way to find R2 score

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('R2 score 2 is {}'.format(r2_1))
print("\n")

# https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.html
