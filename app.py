import streamlit as st
import numpy as np    
import tensorflow as tf
import os,urllib
import librosa # to extract speech features



def main():
    #print(cv2.__version__)
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Emotion Recognition','view source code')
        )
            
    if selected_box == 'Emotion Recognition':        
        st.sidebar.success('To try by yourself by adding a audio file .')
        application()
    if selected_box=='view source code':
        st.code(get_file_content_as_string("app.py"))

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/chiluveri-sanjay/Emotion-recognition/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")
    
@st.cache(show_spinner=False)
def load_model():
    model=tf.keras.models.load_model('mymodel.h5')
    
    return model
def application():
    models_load_state=st.text('\n Loading models..')
    model=load_model()
    models_load_state.text('\n Models Loading..complete')
    
    
    file_to_be_uploaded = st.file_uploader("Choose an audio...", type="wav")
    
    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')
        st.success('Emotion of the audio is  '+predict(model,file_to_be_uploaded))

def extract_mfcc(wav_file_name):
    #This function extracts mfcc features and obtain the mean of each dimension
    #Input : path_to_wav_file
    #Output: mfcc_features'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    
    return mfccs
    
    
def predict(model,wav_filepath):
    emotions={1 : 'neutral', 2 : 'calm', 3 : 'happy', 4 : 'sad', 5 : 'angry', 6 : 'fearful', 7 : 'disgust', 8 : 'surprised'}
    test_point=extract_mfcc(wav_filepath)
    test_point=np.reshape(test_point,newshape=(1,40,1))
    predictions=model.predict(test_point)
    print(emotions[np.argmax(predictions[0])+1])
    
    return emotions[np.argmax(predictions[0])+1]
if __name__ == "__main__":
    main()
