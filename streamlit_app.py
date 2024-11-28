import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("MLB Team Win Predictions")

url = "https://raw.githubusercontent.com/IsabelM8/CS250-FinalProj/refs/heads/main/Teams.csv"
teams = pd.read_csv(url)

# Data preprocessing, transformation
teams = teams[teams['yearID'] > 1960]
teams = teams[~teams['yearID'].isin([1972, 1981, 1994, 1995, 2020])]
teams.drop('divID', axis = 1, inplace = True)
teams.drop('DivWin', axis = 1, inplace = True)
teams.drop('WCWin', axis = 1, inplace = True)
teams['HBP'].fillna(round(teams.groupby('yearID')['HBP'].mean(), 0), inplace = True)
teams['SF'].fillna(round(teams.groupby('yearID')['SF'].mean(), 0), inplace = True)
teams['HBP'].fillna(teams['HBP'].mean(), inplace=True)
teams['SF'].fillna(teams['SF'].mean(), inplace=True)
teams['BA'] = round(teams['H'] / teams['AB'], 3)

# Linear Regression Model
# R2 = 0.89
X = teams[['R', 'BA', 'HR', 'ERA', 'BB', 'SV', 'HA']]
y = teams['W']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
linModel = LinearRegression()
linModel.fit(X_train, y_train)

# Application Function
def predict_wins(r, ba, hr, era, bb, sv, ha):
    pred = linModel.predict([[r, ba, hr, era, bb, sv, ha]])[0]
    return round(pred)

# User Input - Form
with st.form("my_form"):
    st.write("Get your MLB Teams Predicted Wins")
    r = st.number_input("Enter your team's Runs:", format="%.0f")
    ba = st.number_input("Enter your team's Batting Average:", format="%.3f")
    hr = st.number_input("Enter your team's Home Runs:", format="%.0f")
    era = st.number_input("Enter your team's Earned Run Average:", format="%.3f")
    bb = st.number_input("Enter your team's Walks", format="%.0f")
    sv = st.number_input("Enter your team's Saves", format="%.0f")
    ha = st.number_input("Enter your team's Hits Allowed", format="%.0f")

    # Submit Button
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.subheader("Output")
        predictions = predict_wins(r, ba, hr, era, bb, sv, ha)
        st.write(predictions)
