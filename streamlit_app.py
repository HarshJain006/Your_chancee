import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Function to create the DataFrame
def create_dataframe():
    data = {
        'interest_level': [],
        'mutual_friends': [],
        'humor_compatibility': [],
        'common_hobbies': [],
        'communication_freq': [],
        'shared_special_moment': [],
        'will_fall': []
    }
    return pd.DataFrame(data)

# Create a DataFrame
df = create_dataframe()

# User input
st.title("Your Crush Probability Calculator")
interest_level = st.number_input("On a scale of 1 to 10, how much are you interested in your crush?", min_value=1.0, max_value=10.0, step=0.1)
mutual_friends = st.number_input("How many mutual friends do you have?", min_value=0.0, step=1.0)
humor_compatibility = st.number_input("On a scale of 1 to 10, how compatible are you in humor?", min_value=1.0, max_value=10.0, step=0.1)
common_hobbies = st.selectbox("Do you have common hobbies? (0 = No, 1 = Yes)", [0, 1])
communication_freq = st.number_input("On a scale of 1 to 10, how often do you communicate?", min_value=1.0, max_value=10.0, step=0.1)
shared_special_moment = st.number_input("On a scale of 1 to 10, how special is your moment together?", min_value=1.0, max_value=10.0, step=0.1)

# Append user data to DataFrame
df = df.append({
    'interest_level': interest_level,
    'mutual_friends': mutual_friends,
    'humor_compatibility': humor_compatibility,
    'common_hobbies': common_hobbies,
    'communication_freq': communication_freq,
    'shared_special_moment': shared_special_moment,
    'will_fall': None  # Placeholder for prediction
}, ignore_index=True)

# Feature Matrix and Target Vector
X = df[['interest_level', 'mutual_friends', 'humor_compatibility', 'common_hobbies', 'communication_freq', 'shared_special_moment']]
y = df['will_fall'].fillna(0)  # Assuming 0 initially for predictions

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Predict probability
probability = model.predict_proba(X_scaled[-1].reshape(1, -1))[0][1]
st.write(f"Probability your crush will fall for you: {probability * 100:.2f}%")

# Plotting
plt.figure(figsize=(12, 6))
features = ['interest_level', 'mutual_friends', 'humor_compatibility', 'common_hobbies', 'communication_freq', 'shared_special_moment']
values = df.iloc[-1][features]

# Bar plot for probabilities
plt.bar(features, values, color='skyblue')
plt.axhline(y=probability, color='red', linestyle='--', label='Your Probability')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Features", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.title("Crush Probability Features", fontsize=16)
plt.legend()
st.pyplot(plt)
