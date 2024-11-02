import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample data preparation
data = {
    'interest_level': [8.5, 6.2, 9.0, 5.5, 7.8, 6.5, 8.0, 7.2, 6.8, 9.5, 5.0, 8.7, 7.9, 6.1, 8.3],
    'mutual_friends': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    'humor_compatibility': [7.5, 5.0, 9.0, 6.0, 7.8, 5.5, 8.0, 7.0, 6.5, 9.5, 4.0, 8.5, 7.0, 5.5, 8.0],
    'common_hobbies': [2, 0, 3, 1, 0, 1, 2, 2, 0, 3, 1, 2, 1, 0, 3],
    'communication_freq': [4.0, 2.5, 5.0, 3.0, 2.0, 1.5, 4.0, 3.5, 1.0, 5.0, 2.0, 4.5, 3.5, 1.0, 5.0],
    'shared_special_moment': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    'will_fall': [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Define features and target variable
X = df[['interest_level', 'mutual_friends', 'humor_compatibility', 'common_hobbies', 'communication_freq', 'shared_special_moment']]
y = df['will_fall']

# Scaling and model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

st.title("Will Your Crush Fall for You?")

# User input for each feature
interest_level = st.slider("Rate your interest level in conversations (0.0 - 10.0)", 0.0, 10.0, 5.0)
mutual_friends = st.selectbox("Do you have mutual friends?", [1, 0])
humor_compatibility = st.slider("Rate your humor compatibility (0.0 - 10.0)", 0.0, 10.0, 5.0)
common_hobbies = st.number_input("How many common hobbies do you have?", min_value=0, max_value=5, value=1)
communication_freq = st.slider("How often do you chat per week (0.0 - 10.0)", 0.0, 10.0, 5.0)
shared_special_moment = st.selectbox("Have you shared any special moments?", [1, 0])

# Creating the user data input DataFrame
user_data = pd.DataFrame({
    'interest_level': [interest_level],
    'mutual_friends': [mutual_friends],
    'humor_compatibility': [humor_compatibility],
    'common_hobbies': [common_hobbies],
    'communication_freq': [communication_freq],
    'shared_special_moment': [shared_special_moment]
})

# Scale user data
user_data_scaled = scaler.transform(user_data)
probability = model.predict_proba(user_data_scaled)[:, 1][0] * 100

st.write(f"**Probability your crush will fall for you: {probability:.2f}%**")

# Bar plot
features = ['interest_level', 'mutual_friends', 'humor_compatibility', 'common_hobbies', 'communication_freq', 'shared_special_moment']
user_values = user_data.values.flatten()

fig, ax = plt.subplots()
ax.bar(features, user_values, color='lightblue')
ax.axhline(y=probability / 100, color='red', linestyle='--', label='Probability')
plt.xlabel("Features")
plt.ylabel("Values")
plt.title("Your Crush Prediction Analysis")
plt.legend()
st.pyplot(fig)
