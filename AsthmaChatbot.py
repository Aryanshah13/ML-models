#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv(r"C:\Users\shaha\Downloads\processed-data.csv")

# Drop unnecessary columns using column indexing and rename the target column
df = df.drop(df.columns[[4, 5]], axis=1)  # Drop columns by index
df.rename(columns={'Severity_None': 'Target'}, inplace=True)

# Prepare data for training and testing
x = df.drop(columns=['Target'])  # Features
y = df['Target']  # Target variable: 'Target' column represents severity levels

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Classifier with adjusted hyperparameters
model = GradientBoostingClassifier(
    n_estimators=100,  # Increase the number of estimators
    learning_rate=0.1,  # Adjust the learning rate
    max_depth=5,  # Increase the maximum depth of the trees
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)

# Train the model
model.fit(x_train, y_train)

# Function to map "yes" or "no" inputs to integers
def map_yes_no_to_int(answer):
    if answer.lower() == "yes":
        return 1
    elif answer.lower() == "no":
        return 0
    else:
        raise ValueError("Invalid input! Please enter 'yes' or 'no'.")

def check_threshold(symptoms, age, gender):
    # If all symptoms are present and age is greater than or equal to 60, return True
    if all(symptoms) and age >= 60:
        return True
    # If all symptoms are present and age is less than 11, return True
    elif all(symptoms) and age < 11:
        return True
    elif all(symptoms):
        return True
    elif not any(symptoms) and gender == 'male' and age >= 25:
        return False
    else:
        return False

# Function to get recommendation based on severity prediction
def get_recommendation(severity):
    if severity == 0:
        return "Your asthma condition is currently under control. Continue to monitor your symptoms regularly."
    elif severity == 1 or severity == 2:
        return """You are experiencing mild to moderate asthma symptoms. Try some home remedies such as:
            1. Steam Inhalation: Inhale steam from hot water to open up the airways.
            2. Staying Hydrated: Drink plenty of water to keep the airways moist.
            3. Using a Humidifier: Add moisture to the air with a humidifier to prevent dryness in the airways.
            4. Breathing Exercises: Practice deep breathing exercises and pursed-lip breathing to improve lung function.
            5. Avoiding Triggers: Identify and avoid triggers such as smoke, dust, pollen, and pet dander.
            6. Maintaining a Clean Environment: Keep the home clean and free of dust, mold, and allergens."""
    elif severity == 3:
        return """You are experiencing severe asthma symptoms. Please seek immediate medical attention. In the meantime, you may find the following resources helpful:
            1. [How to ease asthma symptoms - 3 effective breathing exercises by Airofit](https://www.youtube.com/watch?v=FyjZLPmZ534)
            2. [Exercise-Induced Asthma by CNN](https://www.youtube.com/watch?v=B8pNeYFZNew)
            3. [ASTHMA / how to cure exercise induced wheezing naturally by Andrew Folts](https://www.youtube.com/watch?v=jv-revgQdPE)
            4. [Easy tips to treat Asthma & Bronchitis | Dr. Hansaji Yogendra by The Yoga Institute](https://www.youtube.com/watch?v=JwRG8AsStLQ)
            5. [Breathing Exercises for COPD, Asthma, Bronchitis & Emphysema - Ask Doctor Jo by AskDoctorJo](https://www.youtube.com/watch?v=dpTNUGwXbTU)"""

# Function to perform prediction
def predict_asthma():
    try:
        name = name_entry.get()
        tiredness = map_yes_no_to_int(tiredness_entry.get())
        dry_cough = map_yes_no_to_int(dry_cough_entry.get())
        difficulty_breathing = map_yes_no_to_int(difficulty_breathing_entry.get())
        sore_throat = map_yes_no_to_int(sore_throat_entry.get())
        nasal_congestion = map_yes_no_to_int(nasal_congestion_entry.get())
        runny_nose = map_yes_no_to_int(runny_nose_entry.get())
        age = int(age_entry.get())
        gender = gender_var.get()

        # Check if all symptoms are "no"
        if tiredness == dry_cough == difficulty_breathing == sore_throat == nasal_congestion == runny_nose == 0:
            # If all symptoms are "no", set severity to 0
            severity_prediction = 0
        else:
            # Check threshold condition
            symptoms = [tiredness, dry_cough, difficulty_breathing, sore_throat, nasal_congestion, runny_nose]
            if check_threshold(symptoms, age, gender):
                severity_prediction = 3  # Set severity to maximum if threshold is met
            else:
                none_experiencing = 1 if sum(symptoms) == 0 else 0
                user_input = [tiredness, dry_cough, difficulty_breathing, sore_throat, nasal_congestion, runny_nose,
                              none_experiencing,  # None_Experiencing feature
                              1 if age <= 9 else 0,  # Age_0-9 feature
                              1 if age >= 10 and age <= 19 else 0,  # Age_10-19 feature
                              1 if age >= 20 and age <= 24 else 0,  # Age_20-24 feature
                              1 if age >= 25 and age <= 59 else 0,  # Age_25-59 feature
                              1 if age >= 60 else 0,  # Age_60+ feature
                              1 if gender == 'female' else 0,  # Gender_Female feature
                              1 if gender == 'male' else 0,  # Gender_Male feature
                              0,  # Severity_Mild placeholder
                              0]  # Severity_Moderate placeholder
                severity_prediction = model.predict([user_input])[0]

        recommendation = get_recommendation(severity_prediction)

        messagebox.showinfo("Prediction Result", f"Hello {name}!\nSeverity Prediction: {severity_prediction}\nRecommendation: {recommendation}\nGender: {gender}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Create GUI
root = tk.Tk()
root.title("Asthma Care Chatbot")

# Load image and resize
chatbot_image = Image.open(r"C:\Users\shaha\Downloads\friendly-chatbot.jpg")
chatbot_image = chatbot_image.resize((100, 100), Image.ANTIALIAS)
chatbot_image = ImageTk.PhotoImage(chatbot_image)
image_label = tk.Label(root, image=chatbot_image)
image_label.image = chatbot_image  # Store image reference to prevent garbage collection
image_label.grid(row=0, column=3)

# GUI components with padding
tk.Label(root, text="Name:").grid(row=1, column=0, padx=10, pady=5)
name_entry = tk.Entry(root)
name_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Tiredness (yes/no):").grid(row=2, column=0, padx=10, pady=5)
tiredness_entry = tk.Entry(root)
tiredness_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Dry Cough (yes/no):").grid(row=3, column=0, padx=10, pady=5)
dry_cough_entry = tk.Entry(root)
dry_cough_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Difficulty in Breathing (yes/no):").grid(row=4, column=0, padx=10, pady=5)
difficulty_breathing_entry = tk.Entry(root)
difficulty_breathing_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Sore Throat (yes/no):").grid(row=5, column=0, padx=10, pady=5)
sore_throat_entry = tk.Entry(root)
sore_throat_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Nasal Congestion (yes/no):").grid(row=6, column=0, padx=10, pady=5)
nasal_congestion_entry = tk.Entry(root)
nasal_congestion_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Runny Nose (yes/no):").grid(row=7, column=0, padx=10, pady=5)
runny_nose_entry = tk.Entry(root)
runny_nose_entry.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Age:").grid(row=8, column=0, padx=10, pady=5)
age_entry = tk.Entry(root)
age_entry.grid(row=8, column=1, padx=10, pady=5)

gender_var = tk.StringVar(value="female")
tk.Label(root, text="Gender:").grid(row=9, column=0, padx=10, pady=5)
gender_female_radio = tk.Radiobutton(root, text="Female", variable=gender_var, value="female")
gender_female_radio.grid(row=9, column=1, padx=10, pady=5)
gender_male_radio = tk.Radiobutton(root, text="Male", variable=gender_var, value="male")
gender_male_radio.grid(row=9, column=2, padx=10, pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_asthma)
predict_button.grid(row=10, columnspan=2, pady=10)

# Run the GUI event loop
root.mainloop()


# In[ ]:




