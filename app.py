import streamlit as st
from fastai.vision.all import *


st.title('COUCH, BED, TABLE prediction...')



import pathlib
import platform
from fastai.vision.core import PILImage


# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
plt = platform.system()
if plt == 'Linux':pathlib.WindowsPath = pathlib.PosixPath



from fastai.learner import load_learner

file = st.file_uploader('Rasim yuklash' , type = ['png','jpeg','jpg','gif','svg'])

if file:
    st.image(file)
    image = PILImage.create(file)
    model = load_learner('moddel_garneture.pkl')

    pred , pred_id , probs = model.predict(image)
    st.success(f"Bashorat:{pred}")
    st.info(f'Ehtimolligi:{probs[pred_id]}')

  

