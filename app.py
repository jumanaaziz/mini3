import streamlit as st
from datetime import datetime
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Function to predict traffic based on the exact time provided by the user
def predict_traffic(day, time_str):
    # Convert the input time string to a time object
    try:
        time = datetime.strptime(time_str, '%H:%M:%S').time()
    except ValueError:
        return "Invalid time format. Please use HH:MM:SS."
    
    # Prepare the input for the model
    day_num = datetime.strptime(day, '%A').weekday()  # Convert day to numerical representation
    hour = time.hour
    minute = time.minute
    
    # Example input for the model with 10 timesteps and 3 features (day_num, hour, minute)
    model_input = np.array([[day_num, hour, minute]] * 10).reshape(1, 10, 3)
    
    # Predict traffic
    prediction = model.predict(model_input)
    
    # Assuming the output is a classification with three categories
    traffic_situation = np.argmax(prediction, axis=1)[0]
    situation_labels = {0: "Low", 1: "Normal", 2: "Heavy"}
    
    # Return a formatted string with prediction
    return f"Predicted traffic for {day} at {time_str} is {situation_labels[traffic_situation]}"

# Streamlit UI
def main():
    st.title("Traffic Prediction System")
    
    # User input for custom prediction
    st.subheader("Custom Traffic Prediction")
    
    day = st.selectbox(
        "Select Day",
        ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
    )
    
    time_str = st.text_input("Enter Time (HH:MM:SS)", value=datetime.now().strftime('%H:%M:%S'))
    
    if st.button("Predict Traffic"):
        custom_prediction = predict_traffic(day, time_str)
        st.write(custom_prediction)

if __name__ == "__main__":
    main()
