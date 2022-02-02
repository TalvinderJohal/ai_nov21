import train as tr
import pandas as pd
import numpy as np
model = tr.train_model()

while True:

    age = int(input("How old are you? \n"))
    sex = str(input("Which is your sex? If you are a male answer 'm', if you are a female answer with an 'f': \n"))
    if sex.lower() == 'm':
        sex = 'male'
    else:
        sex = 'female'
    bmi = float(input("Which is your bmi ('Body mass index')? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = str(input("Do you smoke? Options: yes,no \n"))
    region = str(input("Which is your residential area in the US? Options: northeast, southeast, southwest, northwest \n"))

    #Preprocess
    ct = model[-2]
    scaler = model[-1]
    x = pd.DataFrame({"age":age, "sex":sex, "bmi":bmi, "child":child, "smoke":smoke," region":region}, index=[0])
    x_trans = ct.transform(x)
    x_scaled = scaler.transform(x_trans)

    #Predict

    predictions = np.array([clf.predict(x_scaled) for clf in model[:-2]])

    print(f"Prediction for your conditions: {predictions.mean()}")