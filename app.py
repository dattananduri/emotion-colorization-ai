# ======================================
# Emotion Colorization AI
# ======================================

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Emotion Colorization AI", layout="wide")

# ======================================
# U-NET MODEL
# ======================================

class UNetColorizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.e1 = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2)

        self.e2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2)

        self.e3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d(2)

        self.e4 = nn.Sequential(
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(512,256,4,stride=2,padding=1)
        self.interp1 = nn.Upsample(size=(37,37),mode='bilinear')

        self.conv1 = nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256,128,4,stride=2,padding=1)
        self.interp2 = nn.Upsample(size=(75,75),mode='bilinear')

        self.conv2 = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(128,64,4,stride=2,padding=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv2d(64,2,3,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):

        e1=self.e1(x)
        p1=self.pool1(e1)

        e2=self.e2(p1)
        p2=self.pool2(e2)

        e3=self.e3(p2)
        p3=self.pool3(e3)

        e4=self.e4(p3)

        u1=self.up1(e4)
        u1=self.interp1(u1)
        u1=torch.cat([u1,e3],1)
        d1=self.conv1(u1)

        u2=self.up2(d1)
        u2=self.interp2(u2)
        u2=torch.cat([u2,e2],1)
        d2=self.conv2(u2)

        u3=self.up3(d2)
        u3=torch.cat([u3,e1],1)
        d3=self.conv3(u3)

        return self.out(d3)

# ======================================
# LOAD MODEL
# ======================================

@st.cache_resource
def load_model():

    model = UNetColorizer().to(DEVICE)

    model.load_state_dict(
        torch.load("unet_finetuned_200epochs.pth", map_location=DEVICE)
    )

    model.eval()

    return model

# ======================================
# COLORIZE IMAGE
# ======================================

def colorize_image(model,image):

    img = np.array(image.convert("RGB"))

    h,w = img.shape[:2]

    img_small = cv2.resize(img,(150,150))

    lab = cv2.cvtColor(img_small,cv2.COLOR_RGB2LAB).astype(np.float32)

    L = lab[:,:,0] / 100

    tensor = torch.tensor(L).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)

    pred = pred.cpu()[0].numpy()

    a = np.clip(cv2.resize(pred[0],(w,h))*255,0,255)
    b = np.clip(cv2.resize(pred[1],(w,h))*255,0,255)

    lab_orig = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

    lab_out = np.zeros((h,w,3),dtype=np.uint8)

    lab_out[:,:,0] = lab_orig[:,:,0]
    lab_out[:,:,1] = a.astype(np.uint8)
    lab_out[:,:,2] = b.astype(np.uint8)

    rgb = cv2.cvtColor(lab_out,cv2.COLOR_LAB2RGB)

    # Remove color noise
    rgb = cv2.bilateralFilter(rgb,5,50,50)

    return rgb

# ======================================
# EMOTION FILTER
# ======================================

def emotion_filter(img,brightness,contrast,saturation,warmth):

    img = img.astype(np.float32)

    img += brightness
    img = (img-128)*contrast + 128

    img = np.clip(img,0,255)

    hsv = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[:,:,1] *= saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1],0,255)

    img = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)

    img[:,:,0] *= warmth
    img[:,:,2] *= (2-warmth)

    img = np.clip(img,0,255).astype(np.uint8)

    return img

# ======================================
# EMOTION PRESETS
# ======================================

EMOTIONS = {

"Neutral": (0,1.0,1.0,1.0),

# YOUR HAPPY VALUES
"Happy": (21,1.40,1.40,1.10),

"Sad": (-15,0.8,0.7,0.85),

"Cinematic": (0,1.3,0.9,1.1),

"Vintage": (10,0.8,0.6,1.2),

"Dark": (-20,1.2,0.75,0.95)

}

# ======================================
# UI
# ======================================

st.title("🎨 Emotion Colorization AI")

model = load_model()

uploaded = st.file_uploader("Upload grayscale image",type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded)

    st.image(image,width=400)

    if st.button("Colorize Image"):

        base = colorize_image(model,image)

        st.session_state.base = base

# ======================================
# EMOTION SECTION
# ======================================

if "base" in st.session_state:

    st.subheader("AI Colorized Image")

    st.image(st.session_state.base)

    emotion = st.selectbox("Select Emotion", list(EMOTIONS.keys()))

    b,c,s,w = EMOTIONS[emotion]

    st.subheader("Adjust if Needed")

    brightness = st.slider("Brightness",-100,100,int(b))
    contrast = st.slider("Contrast",0.5,2.0,float(c),0.01)
    saturation = st.slider("Saturation",0.0,3.0,float(s),0.01)
    warmth = st.slider("Warmth",0.5,1.5,float(w),0.01)

    result = emotion_filter(
        st.session_state.base,
        brightness,
        contrast,
        saturation,
        warmth
    )

    st.subheader("Final Output")

    st.image(result)

    st.download_button(
        "Download Image",
        cv2.imencode(".png",result)[1].tobytes(),
        "emotion_output.png"
    )