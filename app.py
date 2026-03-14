# ======================================
# Emotion Colorization AI - With Batch Processing
# ======================================

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd
from io import BytesIO
import zipfile
import time
from pathlib import Path
import tempfile
import os

# REMOVED plotly imports - NOT NEEDED!

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Emotion Colorization AI - Batch Processing", layout="wide")

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
    "Happy": (21,1.40,1.40,1.10),
    "Sad": (-15,0.8,0.7,0.85),
    "Cinematic": (0,1.3,0.9,1.1),
    "Vintage": (10,0.8,0.6,1.2),
    "Dark": (-20,1.2,0.75,0.95)
}

# ======================================
# BATCH PROCESSING FUNCTIONS
# ======================================

def process_batch_images(model, image_list, emotion_params, progress_bar=None):
    """
    Process a batch of images with the same emotion settings
    Returns list of (original, colorized, emotion_applied, filename)
    """
    results = []
    total = len(image_list)
    
    for idx, (img, filename) in enumerate(image_list):
        try:
            # Colorize
            colorized = colorize_image(model, img)
            
            # Apply emotion filter
            b, c, s, w = emotion_params
            emotion_result = emotion_filter(colorized, b, c, s, w)
            
            results.append({
                'original': img,
                'colorized': colorized,
                'emotion': emotion_result,
                'filename': filename
            })
            
            if progress_bar:
                progress_bar.progress((idx + 1) / total)
                
        except Exception as e:
            st.warning(f"Error processing {filename}: {str(e)}")
            continue
    
    return results

def create_zip_of_results(results, include_original=False, include_colorized=True, include_emotion=True):
    """Create a zip file containing all processed images"""
    
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        
        for idx, result in enumerate(results):
            base_name = Path(result['filename']).stem
            
            # Add original if requested
            if include_original and 'original' in result:
                img_bytes = BytesIO()
                result['original'].save(img_bytes, format='PNG')
                zip_file.writestr(f"{base_name}_original.png", img_bytes.getvalue())
            
            # Add colorized (neutral)
            if include_colorized and 'colorized' in result:
                img_bytes = BytesIO()
                Image.fromarray(result['colorized']).save(img_bytes, format='PNG')
                zip_file.writestr(f"{base_name}_colorized.png", img_bytes.getvalue())
            
            # Add emotion-applied
            if include_emotion and 'emotion' in result:
                img_bytes = BytesIO()
                Image.fromarray(result['emotion']).save(img_bytes, format='PNG')
                zip_file.writestr(f"{base_name}_emotion.png", img_bytes.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

def create_batch_report(results, emotion_name, processing_time):
    """Generate a CSV report of batch processing results"""
    
    report_data = []
    for idx, result in enumerate(results):
        report_data.append({
            'Index': idx + 1,
            'Filename': result['filename'],
            'Emotion Applied': emotion_name,
            'Image Size': f"{result['original'].size[0]}x{result['original'].size[1]}",
            'Format': result['filename'].split('.')[-1].upper()
        })
    
    df = pd.DataFrame(report_data)
    
    # Add summary row
    summary = pd.DataFrame([{
        'Index': 'TOTAL',
        'Filename': f'{len(results)} images',
        'Emotion Applied': emotion_name,
        'Image Size': 'Average: ' + str(int(np.mean([r['original'].size[0] for r in results]))),
        'Format': f'Time: {processing_time:.2f}s'
    }])
    
    return pd.concat([df, summary], ignore_index=True)

# ======================================
# MAIN UI WITH BATCH PROCESSING
# ======================================

def main():
    st.title("🎨 Emotion Colorization AI - Batch Processing")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Processing Mode",
            ["Single Image", "Batch Processing"]
        )
        
        st.header("🔬 Analysis Tools")
        show_color_distribution = st.checkbox("Show Color Distribution Analysis", value=False)
        show_metrics = st.checkbox("Show Image Metrics", value=False)
        show_comparison = st.checkbox("Show Comparison Table", value=False)
        
        if mode == "Batch Processing":
            st.header("📦 Batch Settings")
            output_format = st.selectbox("Output Format", ["PNG", "JPG", "All formats"])
            create_zip = st.checkbox("Create ZIP file", value=True)
            include_original = st.checkbox("Include original images in ZIP", value=False)
            include_colorized = st.checkbox("Include colorized (neutral) in ZIP", value=True)
            include_emotion = st.checkbox("Include emotion-applied in ZIP", value=True)
    
    model = load_model()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎨 Colorization", "📦 Batch Processing", "📊 Analysis", "📈 Paper Results"])
    
    with tab1:
        st.header("Single Image Colorization")
        
        uploaded = st.file_uploader("Upload grayscale image", type=["jpg","png","jpeg"], key="single")
        
        if uploaded:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded)
                st.image(image, caption="Original", width=400)
            
            if st.button("Colorize Image"):
                with st.spinner("Colorizing..."):
                    base = colorize_image(model, image)
                    st.session_state.base = base
                    st.session_state.original = image
                    st.rerun()
            
            if "base" in st.session_state:
                with col2:
                    st.image(st.session_state.base, caption="AI Colorized", width=400)
                
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
                st.session_state.result = result
                
                st.download_button(
                    "Download Image",
                    cv2.imencode(".png",cv2.cvtColor(result, cv2.COLOR_RGB2BGR))[1].tobytes(),
                    "emotion_output.png"
                )
    
    with tab2:
        st.header("📦 Batch Image Processing")
        st.markdown("Upload multiple images to process them all with the same emotion settings")
        
        # Batch file uploader
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key="batch"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            # Show preview of uploaded files
            with st.expander("Preview uploaded images"):
                cols = st.columns(5)
                for idx, file in enumerate(uploaded_files[:10]):
                    with cols[idx % 5]:
                        img = Image.open(file)
                        st.image(img, caption=file.name[:15], width=100)
                if len(uploaded_files) > 10:
                    st.write(f"... and {len(uploaded_files) - 10} more")
            
            # Batch processing options
            col1, col2 = st.columns(2)
            
            with col1:
                batch_emotion = st.selectbox(
                    "Apply emotion to all images",
                    list(EMOTIONS.keys()),
                    key="batch_emotion"
                )
                
                # Get preset values
                b_preset, c_preset, s_preset, w_preset = EMOTIONS[batch_emotion]
                
                # Allow adjustment
                st.subheader("Batch Adjustments")
                batch_brightness = st.slider("Brightness", -100, 100, int(b_preset), key="batch_b")
                batch_contrast = st.slider("Contrast", 0.5, 2.0, float(c_preset), 0.01, key="batch_c")
                batch_saturation = st.slider("Saturation", 0.0, 3.0, float(s_preset), 0.01, key="batch_s")
                batch_warmth = st.slider("Warmth", 0.5, 1.5, float(w_preset), 0.01, key="batch_w")
            
            with col2:
                st.subheader("Output Options")
                
                # Preview settings
                show_preview = st.checkbox("Show preview of first image", value=True)
                save_all = st.checkbox("Save all processed images", value=True)
                
                if save_all:
                    save_format = st.selectbox("Save format", ["PNG", "JPG"], key="save_format")
                    naming = st.selectbox(
                        "File naming",
                        ["Original Name + Emotion", "Index + Emotion", "Emotion + Index"]
                    )
            
            # Process batch button
            if st.button("🚀 Process All Images", type="primary"):
                
                # Prepare images list
                image_list = []
                for file in uploaded_files:
                    img = Image.open(file)
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    image_list.append((img, file.name))
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Start timing
                start_time = time.time()
                
                # Process batch
                emotion_params = (batch_brightness, batch_contrast, batch_saturation, batch_warmth)
                
                results = process_batch_images(
                    model, 
                    image_list, 
                    emotion_params,
                    progress_bar
                )
                
                # Calculate time
                processing_time = time.time() - start_time
                
                status_text.success(f"✅ Processed {len(results)} images in {processing_time:.2f} seconds!")
                
                # Store in session state
                st.session_state.batch_results = results
                st.session_state.batch_emotion = batch_emotion
                st.session_state.batch_time = processing_time
                
                # Show preview of first few results
                if show_preview and len(results) > 0:
                    st.subheader("Preview Results")
                    
                    preview_count = min(3, len(results))
                    for i in range(preview_count):
                        st.write(f"**{results[i]['filename']}**")
                        cols = st.columns(3)
                        
                        with cols[0]:
                            st.image(results[i]['original'], caption="Original", use_container_width=True)
                        
                        with cols[1]:
                            st.image(results[i]['colorized'], caption="Neutral Colorized", use_container_width=True)
                        
                        with cols[2]:
                            st.image(results[i]['emotion'], caption=f"✨ {batch_emotion}", use_container_width=True)
                        
                        st.divider()
                
                # Save results
                if save_all:
                    # Create ZIP file
                    zip_buffer = create_zip_of_results(
                        results,
                        include_original=include_original,
                        include_colorized=include_colorized,
                        include_emotion=include_emotion
                    )
                    
                    # Create report
                    report_df = create_batch_report(results, batch_emotion, processing_time)
                    
                    # Display download buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            "📦 Download All as ZIP",
                            zip_buffer,
                            f"emotion_batch_{batch_emotion.lower()}.zip",
                            "application/zip"
                        )
                    
                    with col2:
                        # CSV report
                        csv = report_df.to_csv(index=False)
                        st.download_button(
                            "📊 Download Report CSV",
                            csv,
                            f"batch_report_{batch_emotion.lower()}.csv",
                            "text/csv"
                        )
                    
                    with col3:
                        # Summary stats
                        st.info(f"**Batch Summary**\n\n"
                               f"Images: {len(results)}\n"
                               f"Emotion: {batch_emotion}\n"
                               f"Time: {processing_time:.2f}s\n"
                               f"Speed: {processing_time/len(results):.2f}s/img")
    
    with tab3:
        st.header("📊 Analysis Dashboard")
        
        if "batch_results" in st.session_state:
            results = st.session_state.batch_results
            emotion = st.session_state.batch_emotion
            
            st.subheader(f"Batch Analysis - {emotion}")
            
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", len(results))
            with col2:
                st.metric("Emotion Applied", emotion)
            with col3:
                avg_size = np.mean([r['original'].size[0] for r in results])
                st.metric("Avg Width", f"{int(avg_size)}px")
            with col4:
                st.metric("Processing Time", f"{st.session_state.batch_time:.2f}s")
            
            # Select image for detailed analysis
            selected_idx = st.selectbox(
                "Select image for detailed analysis",
                range(len(results)),
                format_func=lambda x: f"{x+1}. {results[x]['filename']}"
            )
            
            if selected_idx is not None:
                result = results[selected_idx]
                
                tab_orig, tab_neutral, tab_emotion = st.tabs(["Original", "Neutral", f"Emotion ({emotion})"])
                
                with tab_orig:
                    st.image(result['original'], caption="Original Image")
                    
                    if show_color_distribution:
                        with st.spinner("Analyzing original..."):
                            analysis = analyze_color_distribution(result['original'])
                            st.image(analysis['histogram'], caption="Original Color Distribution")
                
                with tab_neutral:
                    st.image(result['colorized'], caption="Neutral Colorization")
                    
                    if show_metrics:
                        metrics = calculate_metrics(result['original'], result['colorized'])
                        cols = st.columns(3)
                        for i, (k, v) in enumerate(metrics.items()):
                            with cols[i % 3]:
                                st.metric(k, f"{float(v):.4f}")
                    
                    if show_color_distribution:
                        with st.spinner("Analyzing neutral..."):
                            analysis = analyze_color_distribution(result['colorized'])
                            st.image(analysis['histogram'], caption="Neutral Color Distribution")
                
                with tab_emotion:
                    st.image(result['emotion'], caption=f"{emotion} Colorization")
                    
                    if show_metrics:
                        metrics = calculate_metrics(result['original'], result['emotion'])
                        cols = st.columns(3)
                        for i, (k, v) in enumerate(metrics.items()):
                            with cols[i % 3]:
                                st.metric(k, f"{float(v):.4f}")
                    
                    if show_color_distribution:
                        with st.spinner("Analyzing emotion..."):
                            analysis = analyze_color_distribution(result['emotion'])
                            st.image(analysis['histogram'], caption=f"{emotion} Color Distribution")
                    
                    # Affective score
                    st.subheader("🎭 Affective Score")
                    score_data = weighted_emotion_score(result['emotion'], emotion)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Affective Score", f"{score_data['score']:.2f}")
                    with col2:
                        st.metric("Red-Green", f"{score_data['a_intensity']:.2f}")
                    with col3:
                        st.metric("Blue-Yellow", f"{score_data['b_intensity']:.2f}")
        
        else:
            st.info("No batch results found. Process some images in the Batch Processing tab first.")
    
    with tab4:
        st.header("📈 Paper Results")
        
        st.subheader("Table 2: Regression Performance")
        table2 = pd.DataFrame({
            "Metric": ["MSE", "MAE", "RMSE", "R²", "MAPE %", "Std Dev"],
            "Training": ["0.001260", "0.000573", "0.000638", "0.987", "0.126", "0.000638"],
            "Validation": ["0.002165", "0.000088", "0.000133", "0.945", "0.217", "0.000133"]
        })
        st.dataframe(table2, use_container_width=True)
        
        if show_comparison:
            st.subheader("Table 3: Comparative Metrics")
            
            data = {
                "Model": ["Baseline", "Traditional", "Standard U-Net", "Emotion U-Net"],
                "MSE": ["0.008450", "0.005230", "0.001450", "0.000579"],
                "PSNR": ["18.45", "20.94", "26.51", "27.82"],
                "SSIM": ["0.721", "0.789", "0.912", "0.934"]
            }
            df3 = pd.DataFrame(data)
            st.dataframe(df3, use_container_width=True)
            
            st.markdown("""
            **Key Findings:**
            - Our Emotion U-Net achieves **27.82 dB PSNR** (best)
            - **0.934 SSIM** shows excellent structural preservation
            - **28.3% improvement** over standard U-Net
            """)

# ======================================
# COLOR DISTRIBUTION ANALYSIS (NO PLOTLY)
# ======================================

def analyze_color_distribution(colorized_image):
    """
    Color Distribution Analysis - Shows color statistics and histograms
    """
    
    # Convert to numpy
    if isinstance(colorized_image, Image.Image):
        img = np.array(colorized_image.convert("RGB"))
    else:
        img = np.array(colorized_image)
    
    # Convert to different color spaces
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rgb = img
    
    # ===== 1. BASIC STATISTICS =====
    stats = {
        "Red Mean": float(np.mean(rgb[:,:,0])),
        "Green Mean": float(np.mean(rgb[:,:,1])),
        "Blue Mean": float(np.mean(rgb[:,:,2])),
        "A (Red-Green) Mean": float(np.mean(lab[:,:,1])),
        "B (Blue-Yellow) Mean": float(np.mean(lab[:,:,2])),
        "Saturation Mean": float(np.mean(hsv[:,:,1])),
        "Value Mean": float(np.mean(hsv[:,:,2])),
    }
    
    # ===== 2. COLOR VARIANCE =====
    variance = {
        "Red Variance": float(np.var(rgb[:,:,0])),
        "Green Variance": float(np.var(rgb[:,:,1])),
        "Blue Variance": float(np.var(rgb[:,:,2])),
        "A Variance": float(np.var(lab[:,:,1])),
        "B Variance": float(np.var(lab[:,:,2])),
    }
    
    # ===== 3. COLOR TEMPERATURE =====
    red_blue_ratio = stats["Red Mean"] / (stats["Blue Mean"] + 1)
    if red_blue_ratio > 1.2:
        temp = "Warm 🌞"
    elif red_blue_ratio < 0.8:
        temp = "Cool ❄️"
    else:
        temp = "Neutral ⚪"
    
    # ===== 4. COLOR RICHNESS =====
    unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
    total_pixels = img.shape[0] * img.shape[1]
    color_percentage = (unique_colors / total_pixels) * 100
    
    # ===== 5. DOMINANT COLOR =====
    red_dom = stats["Red Mean"] > stats["Green Mean"] and stats["Red Mean"] > stats["Blue Mean"]
    green_dom = stats["Green Mean"] > stats["Red Mean"] and stats["Green Mean"] > stats["Blue Mean"]
    blue_dom = stats["Blue Mean"] > stats["Red Mean"] and stats["Blue Mean"] > stats["Green Mean"]
    
    if red_dom:
        dominant = "🔴 Red"
    elif green_dom:
        dominant = "🟢 Green"
    elif blue_dom:
        dominant = "🔵 Blue"
    else:
        dominant = "⚪ Balanced"
    
    # ===== 6. CREATE COLOR HISTOGRAMS =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # RGB Histograms
    colors_rgb = ['red', 'green', 'blue']
    titles = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i, (color, title) in enumerate(zip(colors_rgb, titles)):
        axes[0, i].hist(rgb[:,:,i].ravel(), bins=256, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0, i].set_title(title)
        axes[0, i].set_xlabel('Pixel Value')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
    
    # LAB and HSV Histograms
    axes[1, 0].hist(lab[:,:,1].ravel(), bins=256, color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('A Channel (Red-Green)')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(lab[:,:,2].ravel(), bins=256, color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('B Channel (Blue-Yellow)')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(hsv[:,:,1].ravel(), bins=256, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 2].set_title('Saturation Channel')
    axes[1, 2].set_xlabel('Saturation Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to image
    buf_hist = BytesIO()
    plt.savefig(buf_hist, format='png', dpi=100, bbox_inches='tight')
    buf_hist.seek(0)
    hist_img = Image.open(buf_hist)
    plt.close()
    
    # ===== 7. CREATE COLOR BAR CHART =====
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    channels = ['Red', 'Green', 'Blue']
    means = [stats["Red Mean"], stats["Green Mean"], stats["Blue Mean"]]
    colors_bar = ['red', 'green', 'blue']
    
    bars = ax2.bar(channels, means, color=colors_bar, alpha=0.7)
    ax2.set_title('RGB Channel Means')
    ax2.set_ylabel('Mean Pixel Value')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    buf_bar = BytesIO()
    plt.savefig(buf_bar, format='png', dpi=100, bbox_inches='tight')
    buf_bar.seek(0)
    bar_img = Image.open(buf_bar)
    plt.close()
    
    return {
        "stats": stats,
        "variance": variance,
        "temperature": temp,
        "color_richness": color_percentage,
        "unique_colors": unique_colors,
        "dominant_color": dominant,
        "histogram": hist_img,
        "bar_chart": bar_img
    }

# ======================================
# WEIGHTED AFFECTIVE SCORING
# ======================================

def weighted_emotion_score(image, emotion):
    """
    Paper Formula: Affective Score = w₁·(a - a₀) + w₂·(b - b₀)
    """
    # Convert to numpy array
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = np.array(image)
    
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Get AB channels
    a = lab[:,:,1]
    b = lab[:,:,2]
    
    # Calculate means
    a0 = float(np.mean(a))
    b0 = float(np.mean(b))
    
    # Emotion weights
    emotion_weights = {
        "Neutral": (0.5, 0.5),
        "Happy": (0.7, 0.3),
        "Sad": (0.3, 0.7),
        "Cinematic": (0.8, 0.2),
        "Vintage": (0.4, 0.6),
        "Dark": (0.6, 0.4)
    }
    
    w1, w2 = emotion_weights.get(emotion, (0.5, 0.5))
    
    # Calculate affective score
    affective_score = np.mean(w1 * (a - a0) + w2 * (b - b0))
    
    return {
        "score": float(affective_score),
        "a_intensity": float(np.mean(a)),
        "b_intensity": float(np.mean(b)),
        "dominant": "Red" if np.mean(a) > np.mean(b) else "Blue"
    }

# ======================================
# METRICS CALCULATION
# ======================================

def calculate_metrics(original, colorized):
    """Calculate PSNR, SSIM, MSE, MAE, R²"""
    
    # Convert to numpy arrays
    if isinstance(original, Image.Image):
        orig = np.array(original.convert("RGB"))
    else:
        orig = np.array(original)
    
    if isinstance(colorized, Image.Image):
        col = np.array(colorized.convert("RGB"))
    else:
        col = np.array(colorized)
    
    # Ensure both are 3-channel
    if len(orig.shape) == 2:
        orig = np.stack([orig] * 3, axis=-1)
    if len(col.shape) == 2:
        col = np.stack([col] * 3, axis=-1)
    
    # Get min dimensions
    h = min(orig.shape[0], col.shape[0])
    w = min(orig.shape[1], col.shape[1])
    
    # Resize both
    orig = cv2.resize(orig, (w, h))
    col = cv2.resize(col, (w, h))
    
    # Convert to float
    orig = orig.astype(np.float32)
    col = col.astype(np.float32)
    
    # MSE
    mse = float(np.mean((orig.flatten() - col.flatten()) ** 2))
    
    # MAE
    mae = float(np.mean(np.abs(orig.flatten() - col.flatten())))
    
    # PSNR
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))
    
    # SSIM
    try:
        orig_uint8 = np.clip(orig, 0, 255).astype(np.uint8)
        col_uint8 = np.clip(col, 0, 255).astype(np.uint8)
        ssim = float(structural_similarity(orig_uint8, col_uint8, multichannel=True, win_size=3, channel_axis=-1))
    except:
        try:
            ssim = float(structural_similarity(orig_uint8, col_uint8, multichannel=True))
        except:
            ssim = 0.0
    
    # R²
    ss_res = np.sum((orig.flatten() - col.flatten()) ** 2)
    ss_tot = np.sum((orig.flatten() - np.mean(orig.flatten())) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
    
    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": psnr,
        "SSIM": ssim,
        "R²": r2
    }

if __name__ == "__main__":
    main()
