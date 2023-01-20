import streamlit as st
import torch
import torchvision
import numpy as np
from PIL import Image
import os
import shutil
import random



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://etimg.etb2bimg.com/photo/82210087.cms");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


add_bg_from_url()


st.title("Diagnosis of COVID-19 using PyTorch")

st.write("""### Upload a Chest X-Ray image to predict the disease""")

class_names = ['normal', 'viral', 'covid']


test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(299, 299)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


resnet18 = torchvision.models.inception_v3(pretrained=True)
resnet18.aux_logits=False
resnet18.fc = torch.nn.Linear(in_features=2048, out_features=3)

resnet18.load_state_dict(torch.load('covid_classes (1).pt'))
resnet18.eval()


def predict_image_class(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)
    output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name


uploaded_file = st.file_uploader("", type=['png'])
if uploaded_file is not None:
    image_path = uploaded_file
    st.image(image_path)
    probabilities, predicted_class_index, predicted_class_name = predict_image_class(image_path)
    st.write('Probabilities:', probabilities)
    st.write('Predicted class index:', predicted_class_index)
    st.write('Predicted class name:', predicted_class_name)





