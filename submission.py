import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def feature_eng(df):
    deg_per_rad = 180/math.pi
    # convert timestamp to datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # get geographic wind direction
    df['wind_dir_10'] = 270 - (np.arctan2(-df['U10'],
                               -df['V10']) * deg_per_rad)
    df['wind_dir_100'] = 270 - (np.arctan2(-df['U100'],
                                -df['V100']) * deg_per_rad)
    df['wind_dir_10'] = [dir - 360 if dir > 180 else dir for
                         dir in df['wind_dir_10']]
    df['wind_dir_100'] = [dir - 360 if dir > 180 else dir for
                          dir in df['wind_dir_100']]

    # get wind speeds
    df['wind_spd_10'] = np.sqrt(df['U10']**2 + df['V10']**2)
    df['wind_spd_100'] = np.sqrt(df['U100']**2 + df['V100']**2)

    # create ^2 values
    df['U10_sqr'] = df['U10'] ** 2
    df['V10_sqr'] = df['V10'] ** 2
    df['U100_sqr'] = df['U100'] ** 2
    df['V100_sqr'] = df['V100'] ** 2

    # create month column to user for getting q1-4 groupings
    df['month'] = df['TIMESTAMP'].apply(lambda x: x.month)

    df['q1'] = [1 if month < 4 else 0 for month in df['month']]
    df['q2'] = [1 if 3 < month < 7 else 0 for month in df['month']]
    df['q3'] = [1 if 6 < month < 10 else 0 for month in df['month']]
    df['q4'] = [1 if 9 < month else 0 for month in df['month']]

    # create dummy columns for grouped time of day
    df['day_q1'] = [1 if time < 6 else 0 for
                    time in df['TIMESTAMP'].apply(lambda x: x.hour)]
    df['day_q2'] = [1 if 5 < time < 12 else 0 for
                    time in df['TIMESTAMP'].apply(lambda x: x.hour)]
    df['day_q3'] = [1 if 11 < time < 18 else 0 for
                    time in df['TIMESTAMP'].apply(lambda x: x.hour)]
    df['day_q4'] = [1 if time > 17 else 0 for
                    time in df['TIMESTAMP'].apply(lambda x: x.hour)]

    df = df.drop('month', axis=1)

    return df


def divide_zones(df):
    all_zones = []
    for zone in df['ZONEID'].unique():
        different_zone = df[df['ZONEID'] == zone]
        all_zones.append(different_zone)
    return all_zones


def find_best_regressor(df, regressors, parameters):
    score = 100
    for i in xrange(len(regressors)):
        clf = GridSearchCV(regressors[i], parameters[i],
                           scoring='neg_mean_squared_error',
                           verbose=5, n_jobs=-1)
        X = df.drop(['TARGETVAR', 'TIMESTAMP', 'ZONEID', 'ID'], axis=1)
        y = df['TARGETVAR']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        print "root mean squared error: ", rmse
        if rmse < score:
            score = rmse
            estimator = clf.best_estimator_
            params = clf.best_params_

    return estimator, params


def regressor_per_zone(all_zones):
    rf = RandomForestRegressor()
    gb = GradientBoostingRegressor()

    rf_params = {'n_estimators': [10, 100, 500, 1000], 'n_jobs': [-1],
                 'max_features': [0.5, 0.6, 0.7]}
    gb_params = {'learning_rate': [0.01, 0.1, 0.3, 0.5],
                 'n_estimators': [50, 100, 250, 500],
                 'max_depth': [2, 3, 5, 10]}

    # regressors = [rf, gb, lin]
    # reg_params = [rf_params, gb_params]

    regressors = [rf]
    reg_params = [rf_params]

    zone_estimators = []
    zone_params = []

    for i in xrange(len(all_zones)):
        estimator, params = find_best_regressor(all_zones[i],
                                                regressors, reg_params)
        zone_estimators.append(estimator)
        zone_params.append(params)

    return zone_estimators, zone_params


def train_final_models(all_zones, estimators):
    fully_trained = []
    for i, est in enumerate(estimators):
        X = all_zones[i].drop(['TARGETVAR', 'TIMESTAMP',
                               'ZONEID', 'ID'], axis=1)
        y = all_zones[i]['TARGETVAR']
        est.fit(X, y)
        fully_trained.append(est)

    return fully_trained


def make_predictions(all_zones, estimators):
    zone_predictions = []
    zone_ids = []
    for i in xrange(len(all_zones)):
        X = all_zone_test[i].drop(['TIMESTAMP',
                                   'ZONEID', 'ID'], axis=1).values
        pred = estimators[i].predict(X)
        zone_predictions.extend(pred)
        ids = all_zone_test[i]['ID']
        zone_ids.extend(ids)

    return zone_predictions, zone_ids


if __name__ == "__main__":
    train = pd.read_csv('data/Train_O4UPEyW.csv')
    eng_train = feature_eng(train)
    test = pd.read_csv('data/Test_uP7dymh.csv')
    eng_test = feature_eng(test)

    all_zone_train = divide_zones(eng_train)
    all_zone_test = divide_zones(eng_test)

    estimators, parameters = regressor_per_zone(all_zone_train)

    fully_trained_models = train_final_models(all_zone_train, estimators)

    full_pred, full_ids = make_predictions(all_zone_test, fully_trained_models)

    results = pd.DataFrame(full_pred, full_ids)
    results.to_csv('predictions_gs.csv')

    # simplistic model limited GridSearchCV
    # rf_regressors = []
    # for i in xrange(len(all_zone_train)):
    #     rf = RandomForestRegressor(n_estimators=500, verbose=3,
    #                                max_features=0.6, n_jobs=-1)
    #     X = all_zone_train[i].drop(['TARGETVAR', 'TIMESTAMP',
    #                                 'ZONEID', 'ID'], axis=1)
    #     y = all_zone_train[i]['TARGETVAR']
    #     rf.fit(X, y)
    #     rf_regressors.append(rf)
    #
    # zone_predictions = []
    # zone_ids = []
    # for i in xrange(len(all_zone_test)):
    #     X = all_zone_test[i].drop(['TIMESTAMP',
    #                                'ZONEID', 'ID'], axis=1).values
    #     pred = rf_regressors[i].predict(X)
    #     zone_predictions.extend(pred)
    #     ids = all_zone_test[i]['ID']
    #     zone_ids.extend(ids)
    #
    # print len(zone_ids), len(zone_predictions)
    #
    # results = pd.DataFrame(zone_predictions, zone_ids)
    # results.to_csv('prediction_results.csv')


"""
bottom of page
"""
