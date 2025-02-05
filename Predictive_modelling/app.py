
import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('final_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('transformer.pkl', 'rb') as file:
    pt = pickle.load(file)

def prediction(input_list):

    input_list = np.array(input_list, dtype = object)

    pred = model.predict_proba([input_list])[:,1][0]

    if pred > 0.5:
        return f'''You Have More Chances of Getting Diseased
Your Probability Of Having Cardio Vascular Disease is {round(pred,2)}
Take Care'''

    else:
        return f'''You Have less Chances of Getting Diseased
Your Probability Of Having Cardio Vascular Disease is {round(pred,2)}
Take Care'''



def main():
    st.title('CardioLens : Analyzing lifestyle data to predict heart disease risks')

    h = st.text_input('What is Your Height (In cm) ?')

    w = st.text_input('What is Your Weight (in Kg) ?')

    gh = (lambda x:0 if x=='Poor' else 1 if x == 'Fair' else 2 if x=='Good' else 3 if x=='Very Good' else 4)\
        (st.selectbox('What About Your General Health ?', ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']))

    bg = (lambda x: 0 if x=='O' else 1 if x=='B' else 2 )(st.selectbox('What is Your Blood Group ?', ['A', 'B', 'O', 'AB']))

    age = (lambda x: 0 if x < 30 else 1 if x < 50 else 2)(st.slider('Tell us How old are you ? ', min_value = 18, max_value = 100, step = 1))

    bmi = st.text_input('Tell us about Your Body Mass Index ?')

    ckup = (lambda x: 0 if x=='Never' else 1 if x=='Within the past year' else 2 if 
    x=='Within the past 2 years' else 3 if x=='Within the past 5 years' else 4)(st.selectbox("When was the last time you had a check-up?"\
        ,['Never','Within the past year', 'Within the past 2 years', 'Within the past 5 years']))

    smk = (lambda x: 0 if x=='No' else 1)(st.selectbox('Do You Smoke ?', ['Yes', "No"]))

    sex = (lambda x: 0 if x=='Female' else 1)(st.selectbox('Tell Us About Your Gender ?', ['Male', "Female"]))

    dp = (lambda x: 0 if x=='Non Veg' else 1)(st.selectbox('Tell Us About Your Diet Preference ?', ['Veg', "Non Veg"]))

    gv = st.slider('Rate Your Green Vegetable Consumption (where low is 1 and high is 4)', min_value = 1, max_value = 4, step = 2)

    fr = st.slider('Rate Your Fruit Consumption (where low is 1 and high is 4)', min_value = 1, max_value = 4, step = 2)

    fry = st.slider('Rate Your Fried Food Consumption (where low is 1 and high is 4)', min_value = 1, max_value = 4, step = 2)

    alco = st.slider('Rate Your Alcohol Consumption (where low is 1 and high is 4)', min_value = 1, max_value = 4, step = 2)

    dep = (lambda x: 0 if x=='No' else 1)(st.selectbox('Are You Feeling Depressed ?', ['Yes', "No"]))

    mar = (lambda x: 0 if x=='Married' else 1)(st.selectbox('What About Your Marital Status ?', ['Married', "UnMarried"]))

    exer = (lambda x: 0 if x=='No' else 1)(st.selectbox("Do you work out every day?", ['Yes', "No"]))

    skin = (lambda x: 0 if x=='No' else 1)(st.selectbox("Have You Ever Diagnosed With Skin Cancer ?", ['Yes', "No"]))

    other = (lambda x: 0 if x=='No' else 1)(st.selectbox("Have You Ever Diagonsed Wth ANy Type Of Cancer?", ['Yes', "No"]))

    arth = (lambda x: 0 if x=='No' else 1)(st.selectbox("Are You Suffering from Arthritis ?", ['Yes', "No"]))

    diab = (lambda x: 0 if x=='No' else 1)(st.selectbox("Do You Have Any Type Of Diabetes?", ['Yes', "No"]))

    vac = (lambda x: 0 if x=='No' else 1)(st.selectbox("Are You Vaccinated Aginst Cardio Vascular Disease ?", ['Yes', "No"]))

    tran_data = pt.transform([[h,w,bmi]])

    h_t = tran_data[0][0]

    w_t = tran_data[0][1]

    bmi_t = tran_data[0][2]


    input_list = [h_t, w_t, gh, bg, age, bmi_t, ckup, smk, sex, dp, gv, fr, fry, alco, dep, mar, exer, skin, other, arth, diab, vac]


    if st.button('Show Prediction'):
        response = prediction(input_list)
        st.success(response)


if __name__ == '__main__':
    main()


                                        
    
    

