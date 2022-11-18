#Core Pkgs
import streamlit as st
import altair as alt

# EDA Pkgs
import pandas as pd
import numpy as np

#Utils
import joblib

#Function to load model
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr.pkl","rb"))

#Function to predict
def predict_emotion(docx):
    result = pipe_lr.predict([docx])
    return result[0]

def predict_prob_emotion(docx):
    result = pipe_lr.predict_proba([docx])
    return result

def main():
    st.title("Emotion Classifier ML App")
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Enter Text")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            prediction = predict_emotion(raw_text)
            prediction_probibility = predict_prob_emotion(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence: {}".format(np.max(prediction_probibility)))
            
            with col2:
                st.success("Prediction Probability")
                prob_df = pd.DataFrame(prediction_probibility,columns=pipe_lr.classes_)
                clean_df = prob_df.T.reset_index()
                clean_df.columns = ['Emotion','Probability']
                
                fig = alt.Chart(clean_df).mark_bar().encode(
                    x='Emotion',
                    y='Probability',
                    color = 'Emotion'
                )
                st.altair_chart(fig,use_container_width=True)


    
    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")  




if __name__ == '__main__':
    main()
