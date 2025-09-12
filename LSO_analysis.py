## -----------------------------------------------------------------------------
##
## Script name: LSO_analysis.R
##
## Purpose of script: Perform Leave-Site-Out cross validation of the LightGBM model framework for
## predicting daily dead fuel moisture content at 27 sites across Victoria, Australia.
##
## Author: Tom Keeble
##
## Date Created: 2025-09-12
##
## Copyright (c) Tom Keeble, 2025
## Email: tom.keeble0@unimelb.edu.au
##
## -----------------------------------------------------------------------------
##
## Notes: Without both predictor and predictand data, as well as the trained models (both subject to data sharing limitations), this script 
## is not usable, but is provided as demonstration of the approach.
##   
## -----------------------------------------------------------------------------

### 1. MODULES AND CONSTANTS

import csv
import copy
import pandas as pd
import numpy as np
from joblib import load

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

### 1b. Set constants
stats = ['min']
DAYS_TO_PREDICT = 7
INPUT_PATH = '/path/to/data/'


###############################################################################################
### 2a. DATA

# Read in the X_ (predictor) and y_data (predictand; the daily dmfc observations)
# min, across each day to predict (7 days)
X_data = {}; y_data = {}
for stat in stats:
    X_data[stat] = {}; y_data[stat] = {}; 
    for dyp in range( DAYS_TO_PREDICT ):
        jlname_x = INPUT_PATH + 'X_data_%s_day%s.joblib'  % ( stat, dyp )
        jlname_y = INPUT_PATH + 'y_data_%s_day%s.joblib'  % ( stat, dyp )

        X_data[stat][dyp] = load( jlname_x )
        y_data[stat][dyp] = load( jlname_y )
        print(f"Loaded day {dyp}")
print("Loaded complete input dataset joblibs")


### 2b. Convert to pd.dataframe for ease of use (slower though)

for stat in stats:
    for dyp in range(DAYS_TO_PREDICT):
        X_data[stat][dyp] = pd.DataFrame(X_data[stat][dyp])
        y_data[stat][dyp] = pd.DataFrame(y_data[stat][dyp])
        print(f"dataframed day {dyp}")


### 2c. Extract site elevations to use as unique identifier in LSO
site_elevs = np.unique(X_data['min'][0].iloc[:, 2])


###############################################################################################
### 3a. Create CSV and write header for results
with open('/path/to/analytics/LSO_results.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Elev', 'Stat', 'Day', 'Val_RMSE', 'LSO_RMSE'])

### 3b. Function to write out summary results to CSV
def write_csv_summary(data):
    with open('/path/to/analytics/LSO_results.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


###############################################################################################
### 4. Leave Site Out (LSO) cross validation across all 27 sites
### (results from these are summarised across all sites in Figure 3 of the manuscript, and Figure 4 shows an example timeseries forecast for a single left-out site)
for site in site_elevs:
    print('Running site removal for site with elevation: ', site)

    x_df_less_site = {}; y_df_less_site = {}
    x_df_site = {}; y_df_site = {}
    lgb_mod = {}
    for stat in stats:
        x_df_less_site[stat] = {};  y_df_less_site[stat] = {}
        x_df_site[stat] = {};  y_df_site[stat] = {}
        lgb_mod[stat] = {}; #lgb_l[stat] = {}; lgb_u[stat] = {}; 
        for dyp in range( DAYS_TO_PREDICT ):
            mask = X_data[stat][dyp].iloc[:, 2].values != site # Boolean mask of site based on unique elevation value
            x_df_less_site[stat][dyp] = X_data[stat][dyp][mask] # Apply mask
            x_df_site[stat][dyp] = X_data[stat][dyp][~mask]

            y_df_less_site[stat][dyp] = y_data[stat][dyp][mask] # Apply mask
            y_df_site[stat][dyp] = y_data[stat][dyp][~mask]

            lgb_mod[stat][dyp] = LGBMRegressor( objective = 'regression', learning_rate=0.05, n_estimators=10000, silent = False, feature_fraction = 0.9, max_depth=-1, num_leaves=31)
            #lgb_l[stat][dyp] = LGBMRegressor( objective='quantile', metric='quantile', alpha=0.1 )  # 10% confidence interval
            #lgb_u[stat][dyp] = LGBMRegressor( objective='quantile', metric='quantile', alpha=0.9 )  # 90% confidence interval
    print('Removed site from training data')



    X_train = {}; y_train = {}
    X_val = {}; y_val = {}
    for stat in stats:
        X_train[stat] = {}; y_train[stat] = {}
        X_val[stat] = {}; y_val[stat] = {};
        for dyp in range( DAYS_TO_PREDICT ):
            X_train[stat][dyp], X_val[stat][dyp], y_train[stat][dyp], y_val[stat][dyp] = train_test_split(x_df_less_site[stat][dyp], 
                                                                                                        y_df_less_site[stat][dyp], 
                                                                                                        test_size=0.1, 
                                                                                                        random_state=42, stratify = x_df_less_site[stat][dyp].iloc[:, 2])

            lgb_mod[stat][dyp].fit(X_train[stat][dyp], y_train[stat][dyp], 
                eval_set=(X_val[stat][dyp], y_val[stat][dyp]), 
                early_stopping_rounds=1000, eval_metric='rmse')
            
            y_mod_val_pred = lgb_mod[stat][dyp].predict(X_val[stat][dyp], num_iteration=lgb_mod[stat][dyp].best_iteration_)
            # Evaluate the model on the validation set
            rmse_val = mean_squared_error(y_val[stat][dyp], y_mod_val_pred, squared = False)
            print(f"Mean Squared Error of LGBM API day-{dyp} model on Validation Set: {rmse_val}")

            y_mod_site_pred = lgb_mod[stat][dyp].predict(x_df_site[stat][dyp], num_iteration=lgb_mod[stat][dyp].best_iteration_)
            # Evaluate the model on the validation set
            rmse_site = mean_squared_error(y_df_site[stat][dyp], y_mod_site_pred, squared = False)
            print(f"Mean Squared Error of LGBM API day-{dyp} model on left-out site dataset: {rmse_site}")

            write_csv_summary([site, stat, dyp, rmse_val, rmse_site])

            pred_obs = pd.DataFrame({'DayOfYear': list(x_df_site[stat][dyp].iloc[:,9]),
                                    'Year': list(x_df_site[stat][dyp].iloc[:,10]),
                                    'Obs': list(y_df_site[stat][dyp].iloc[:,0]),
                                    'Preds': list(y_mod_site_pred)
                                    })

            pred_obs.to_csv('/path/to/analytics/LSO_%f_%s_%s.csv' % ( site, stat, dyp ), index = False) 
        
