import Trainer as tr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
model = tr.train_model()

while True:

    age = int(input("How old are you? \n"))
    sex = str(input("Which is your sex? If you are a male answer 'm', if you are a female answer with an 'f': \n "))
    if sex.lower() == 'm':
        sex = 1
    else:
        sex = 0
    cp = int(input("What type of chest pain are you having? Answer with a integer betwee 0 - 3: \n" ))
    trtbps = int(input("What is ure resting blood pressure (in mm Hg):  \n"))
    chol = int(input("What is ure cholestrol level in mg/dl?:   \n"))
    fbs = int(input("Is your fasting blood sugar greater than 120mg/dl? Answer Yes as 1 and No as 0:    \n"))
    restecg = int(input("What are your resting electrocardiographic results?Answer with a integer between 0 - 2:  \n"))
    thalachh = int(input("What is ure maximum heart rate you achieve?   \n"))
    exng = int(input("Does exercise induce Engina?: Yes = 1 and No = 0: \n"))
    oldpeak = float(input("What is ure previous peak? \n"))
    slope = int(input("What is ure slope? Answer with a integer 0 - 2:   \n"))
    caa = int(input("What are the number of major vessels?Answer with a integer between 0 - 4  \n"))
    thall = int(input("What is your thall rate?Answer with a integer between 0 - 3:  \n"))

    scaler = model[-1]
    x = pd.DataFrame({"age":age, "sex":sex, "chest pain":cp, "resting bp":trtbps, "cholestrol":chol, "fasting blood sugar":fbs,
                    "resting electrocardiographic results":restecg, "max heart rate":thalachh, "exercise induced engine":exng, "previous peak":oldpeak,
                     "slope":slope, "major vessels":caa, "thal rate":thall}, index=[0])
    x_scaled = scaler.transform(x)

    predictions = np.array([clf.predict(x_scaled) for clf in model[:-1]])
    print(f"Prediction for your condition is: {predictions.mean()} (1 = You are most susceptible to a Heart attack, 0 = YOu are not susceptible to a Heart attack)")

    # 63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1