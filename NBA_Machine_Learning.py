import pandas as pd
import tensorflow as tf
import numpy as np
df = pd.read_csv("nba_games.csv", index_col=0)

# re-ordering data
df = df.sort_values("date")
df = df.reset_index(drop=True)

# cleaning data
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]

def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)

# setting a target that means target is next game won 1 or 0
df["target"][pd.isnull(df["target"])] = 2
df["target"] = df["target"].astype(int, errors="ignore")
# just removing all null=2 values in target
df = df[df["target"] != 2]
# doing the same for won
df["won"] = df["won"].astype(int, errors="ignore")


df["won"].value_counts()
df["target"].value_counts()

# checking all Null values
nulls = pd.isnull(df).sum()
nulls = nulls[nulls >0]

# removing all Null values in df
valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns].copy()
# print(df.head())

# columns to not normalize
remove_columns = ["season", "date", "won", "target", "team", "team_opp"]
# using the negation operator(~) to exclude the not normalize columns 
selected_columns = df.columns[~df.columns.isin(remove_columns)]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

# Getting Predictors(labels X) for model
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

rr = RidgeClassifier(alpha=1)
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr,
                                n_features_to_select=54,
                                direction="forward",
                                cv=split,
                                n_jobs=1
                                )
sfs.fit(df[selected_columns], df["target"])
predictors = list(selected_columns[sfs.get_support()])
#print(predictors)

# Rolling avarages for each team

df_rolling = df[list(selected_columns) +["won", "team", "season"]]

# Finding avarages function
def find_team_averages(team):
    rolling = team.rolling(10).mean()
    return rolling

# Grouping df by team and season
df_rolling = df_rolling.groupby(["team","season"], group_keys=False).apply(find_team_averages)
# Renaming these new avarages columns
rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols

# Unite to df
df = pd.concat([df, df_rolling], axis=1)
df = df.dropna() # Dropping all NaN values

# Making a better Predictor by knowing who is home and away by team
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col
def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x,col_name))

df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df,"date")
df = df.copy()

# Pulling opponent same data of next game home or away
full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], 
                left_on=["team","date_next"],
                right_on=["team_opp_next", "date_next"])
#print(full[["team_x", "team_opp_next_x", "team_y", "team_opp_next_y", "date_next"]])

# Finding new predictor
# Columns we do not want
remove_columns = list(full.columns[full.dtypes == "object"]) + remove_columns

selected_columns = full.columns[~full.columns.isin(remove_columns)]

sfs.fit(full[selected_columns], full["target"])

predictors = list(selected_columns[sfs.get_support()])

print(predictors)

# Trying train_test_split from Sklearn.model
from sklearn.model_selection import train_test_split

# Spliting the data into train and test sets try full[predictors] later as x
x_train, x_test, y_train, y_test = train_test_split(full[predictors], full["target"], test_size=0.2, random_state=42)


# define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(54, input_shape=(x_train.shape[1],),
activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation= 'softmax'))


# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])

# convert the target to categorical

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model.fit(x_train, y_train, epochs= 100, batch_size=45)

# Save the model
model.save("trained_model.h5")

# Use the test data to evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Make predictions on the test data
test_predictions = model.predict(x_test)
print(test_predictions)

team1 = "GSW" #home
team2 = "CHI" #away

team_1_rows = full.loc[(full["team_x"] == team1)]
team_2_rows = full.loc[(full["team_y"] == team2)]

matchup = pd.DataFrame(columns=predictors)

matchup.loc[0] = team_1_rows[predictors].iloc[-1]
matchup.loc[1] = team_2_rows[predictors].iloc[-1]

matchup_predictions = matchup[predictors]
print(matchup_predictions)

matchup_predictions = model.predict(matchup_predictions)

# Print the predictions
print(f'Prediction for {team1}: {matchup_predictions[0]}')
print(f'Prediction for {team2}: {matchup_predictions[1]}')
