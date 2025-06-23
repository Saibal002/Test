import streamlit as st 
import cv2
import numpy as np
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import io

# --------------------------------
# Custom CSS for Modern UI
# --------------------------------
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50 !important;
    }
    
    .sidebar .sidebar-content {
        color: white;
    }
    
    /* Title styling */
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    
    h2, h3, h4 {
        color: #34495e;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* File uploader styling */
    .stFileUploader>div>div>div>div {
        border: 2px dashed #3498db;
        border-radius: 8px;
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    /* Radio button styling */
    .stRadio>div>label>div {
        padding: 8px 16px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stRadio>div>label>div:hover {
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    /* Card-like containers */
    .card {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 20px;
        background-color: white;
        margin-bottom: 20px;
    }
    
    /* Success message */
    .success {
        color: #27ae60;
        font-weight: bold;
    }
    
    /* Warning message */
    .warning {
        color: #f39c12;
        font-weight: bold;
    }
    
    /* Error message */
    .error {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Config
# --------------------------------
YOLO_MODEL_PATH = "best.pt"
CNN_MODEL_PATH = "ocr_model_kaggle.keras"
DETECTED_FOLDER = "detected_regions/"

os.makedirs(DETECTED_FOLDER, exist_ok=True)

# --------------------------------
# Load models
# --------------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)
ocr_model = CNN_MODEL_PATH

CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_int = {char: i for i, char in enumerate(CHARACTERS)}
int_to_char = {i: char for char, i in char_to_int.items()}

# --------------------------------
# Helper functions
# --------------------------------
def show_image(title, image, cmap='gray'):
    buf = io.BytesIO()
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    st.image(buf.getvalue(), caption=title)

def detect_plates(image_np, image_name):
    results = yolo_model(image_np)
    boxes = results[0].boxes.xyxy
    plates = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = image_np[y1:y2, x1:x2]
        path = os.path.join(DETECTED_FOLDER, f"{image_name}_plate_{i}.jpg")
        cv2.imwrite(path, crop)
        plates.append((crop, (x1, y1, x2, y2)))

    return plates, results[0].plot()

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upsampled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(upsampled)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    show_image("Enhanced Image", sharpened)
    return sharpened

def adaptive_threshold(img):
    mean_intensity = np.mean(img)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mean_intensity < 127:
        binary = cv2.bitwise_not(binary)
    show_image("Binarized Image", binary)
    return binary

def segment_characters(image):
    enhanced = enhance_image(image)
    resized = cv2.resize(enhanced, (333, 75))
    binary = adaptive_threshold(resized)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)
    show_image("Post Morphology", binary)

    LP_WIDTH, LP_HEIGHT = binary.shape
    binary[:3, :] = 255
    binary[:, :3] = 255
    binary[LP_WIDTH-3:, :] = 255
    binary[:, LP_HEIGHT-3:] = 255

    cntrs, _ = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    debug = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug, cntrs, -1, (0, 255, 255), 1)
    show_image("Contours", debug, cmap=None)

    char_imgs, bboxes, centroids = [], [], []
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    for cnt in cntrs:
        x, y, w, h = cv2.boundingRect(cnt)
        if 15 < w < 100 and 20 < h < 100:
            char = binary[y:y+h, x:x+w]
            char = cv2.copyMakeBorder(char, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
            resized_char = cv2.resize(char, (32, 32))
            char_imgs.append(resized_char)
            bboxes.append((x, y, w, h))
            centroids.append((x + w//2, y + h//2))

    if not char_imgs:
        return [], [], debug 

    cy_range = max(c[1] for c in centroids) - min(c[1] for c in centroids)
    if cy_range < 20:
        sorted_chars = sorted(zip(bboxes, char_imgs), key=lambda b: b[0][0])
    else:
        median_cy = np.median([cy for _, cy in centroids])
        top = [i for i, (_, cy) in enumerate(centroids) if cy < median_cy]
        bottom = [i for i in range(len(char_imgs)) if i not in top]
        sorted_chars = sorted([(bboxes[i], char_imgs[i]) for i in top], key=lambda b: b[0][0]) + \
                       sorted([(bboxes[i], char_imgs[i]) for i in bottom], key=lambda b: b[0][0])

    bboxes = [item[0] for item in sorted_chars]
    char_imgs = [item[1] for item in sorted_chars]

    return char_imgs, bboxes, debug
def correct_plate_text(pred):
    STATE_CODES = {
        'AN': 'Andaman and Nicobar Islands', 'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh',
        'AS': 'Assam', 'BR': 'Bihar', 'CG': 'Chhattisgarh', 'CH': 'Chandigarh', 'DD': 'Daman and Diu',
        'DL': 'Delhi', 'DN': 'Dadra and Nagar Haveli', 'GA': 'Goa', 'GJ': 'Gujarat', 'HP': 'Himachal Pradesh',
        'HR': 'Haryana', 'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir', 'KA': 'Karnataka', 'KL': 'Kerala',
        'LA': 'Ladakh', 'LD': 'Lakshadweep', 'MH': 'Maharashtra', 'ML': 'Meghalaya', 'MN': 'Manipur',
        'MP': 'Madhya Pradesh', 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'PB': 'Punjab',
        'PY': 'Puducherry', 'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TR': 'Tripura',
        'TS': 'Telangana', 'UK': 'Uttarakhand', 'UP': 'Uttar Pradesh', 'WB': 'West Bengal'
    }

    COMMON_CONFUSIONS = {
        'O': '0',
        'Q': '0', 
        'D': '0',
        '0': 'O',
        'I': '1',
        'L': '4',
        'T': '1',
        '1': 'I',
        'Z': '2',
        '2': 'Z', 
        'S': '5', 
        '5': 'S',
        'B': '8',
        '8': 'B', 
        'G': '0', 
        'G': 'C',
        '6': 'G',
        'J': '3',
    }

    pred = pred.strip().upper()
    original = pred

    if not (9 <= len(pred) <= 10):
        print(f"❌ Invalid length: {pred}")
        return pred

    chars = list(pred)
    pattern = ['A', 'A', 'N', 'N', 'A', 'A', 'N', 'N', 'N', 'N'] if len(chars) == 10 else ['A', 'A', 'N', 'N', 'A', 'N', 'N', 'N', 'N']

    for i in range(len(chars)):
        expected = pattern[i]
        c = chars[i]
        if expected == 'A' and not c.isalpha():
            corrected = COMMON_CONFUSIONS.get(c, 'A')
            print(f"🔠 Position {i+1}: {c} → {corrected} (expected letter)")
            chars[i] = corrected
        elif expected == 'N' and not c.isdigit():
            corrected = COMMON_CONFUSIONS.get(c, '0')
            print(f"🔢 Position {i+1}: {c} → {corrected} (expected digit)")
            chars[i] = corrected

    c1, c2 = chars[0], chars[1]
    state_code = c1 + c2

    if state_code not in STATE_CODES:
        if c2 == 'B':
            if c1 in {'H', 'M', 'N'}:
                chars[0] = 'W'; state_code = 'WB'
                print(f"⚠️ State code {c1+c2} invalid → correcting to WB")
            elif c1 not in {'P', 'W'}:
                chars[0] = 'P'; state_code = 'PB'
                print(f"⚠️ State code {c1+c2} invalid → correcting to PB")
        elif c1 == 'D':
            if c2 in {'0', 'O', 'U'}:
                chars[1] = 'D'; state_code = 'DD'
                print(f"⚠️ State code {c1+c2} likely → DD (Daman and Diu)")
            elif c2 in {'1', '2', 'I', 'Z'}:
                chars[1] = 'L'; state_code = 'DL'
                print(f"⚠️ State code {c1+c2} likely → DL (Delhi)")
            elif c2 in {'W', 'V', 'M', 'N'}:
                chars[1] = 'N'; state_code = 'DN'
                print(f"⚠️ State code {c1+c2} likely → DN (Dadra and Nagar Haveli)")
        elif c2 == 'P' and c1 in {'N', 'H'}:
            chars[0] = 'M'; state_code = 'MP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to MP")
        elif c2 == 'H' and c1 in {'N', 'W'}:
            chars[0] = 'M'; state_code = 'MH'
            print(f"⚠️ State code {c1+c2} invalid → correcting to MH")
        elif c2 == 'P' and c1 in {'V', 'O'}:
            chars[0] = 'U'; state_code = 'UP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to UP")
        elif c1 == 'A' and c2 != 'P':
            chars[1] = 'P'; state_code = 'AP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to AP")
        elif c1 == 'T' and c2 != 'N':
            chars[1] = 'N'; state_code = 'TN'
            print(f"⚠️ State code {c1+c2} invalid → correcting to TN")
        elif c1 == 'G' and c2 not in {'J'}:
            chars[1] = 'J'; state_code = 'GJ'
            print(f"⚠️ State code {c1+c2} invalid → correcting to GJ")
        elif c2 == 'P' and c1  in {'W','V'}:
            chars[1] = 'MP'; state_code = 'MP'
            print(f"⚠️ State code {c1+c2} invalid → correcting to MP")    

    corrected = ''.join(chars)

    if corrected != original:
        print(f"🔁 Predicted: {original} → Corrected: {corrected}")
    else:
        print(f"✅ Plate Format Valid: {corrected}")

    state_code = corrected[:2]
    state_name = STATE_CODES.get(state_code, "Unknown State Code")
    print(f"🌐 Vehicle registered in: {state_name} ({state_code})")

    return corrected, state_name

def pad_and_prepare(img):
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_plate(chars, bboxes, plate_img):
    prediction = ""
    debug_img = plate_img.copy()
    for char_img, bbox in zip(chars, bboxes):
        proc = pad_and_prepare(char_img)
        pred = ocr_model.predict(proc, verbose=0)
        label = int_to_char[np.argmax(pred)]
        prediction += label
        x, y, w, h = bbox
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    show_image("Final Prediction", debug_img, cmap=None)
    
    corrected_prediction = correct_plate_text(prediction)
    return prediction,corrected_prediction

# --------------------------------
# Streamlit UI - Modern Version
# --------------------------------
st.title("🚗 Vehicle License Plate Recognition System")
st.markdown("""
<div class="card">
    <p>This application detects and recognizes license plates using YOLO for detection and a CNN for character recognition.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    option = st.radio("Select Mode", ["Upload Images", "Live Webcam Detection"])
    st.markdown("---")
    st.markdown("""
    <div style="color: white; font-size: small;">
        <p><strong>Developed by:</strong> Your Name</p>
        <p><strong>College Project</strong></p>
    </div>
    """, unsafe_allow_html=True)

if option == "Upload Images":
    st.subheader("📤 Upload Images")
    uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"🔍 Processing: {uploaded_file.name}", expanded=True):
                image_name = uploaded_file.name.split('.')[0]
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                st.markdown("### 🔎 Detection Results")
                plates, annotated = detect_plates(image_bgr, image_name)
                st.image(annotated, caption="Detected Plates", use_column_width=True)

                if not plates:
                    st.markdown('<p class="warning">⚠️ No license plates detected in this image.</p>', unsafe_allow_html=True)
                else:
                    for idx, (plate_img, box) in enumerate(plates):
                        with st.container():
                            st.markdown(f"### 🚘 Plate {idx + 1} Analysis")
                            cols = st.columns(2)
                            with cols[0]:
                                st.image(plate_img, caption=f"Plate Region {idx + 1}", use_column_width=True)
                            
                            with cols[1]:
                                with st.spinner("Processing characters..."):
                                    chars, bboxes, segmented_debug = segment_characters(plate_img)
                                    if not chars:
                                        st.markdown('<p class="warning">⚠️ No characters found in this plate.</p>', unsafe_allow_html=True)
                                    else:
                                        raw_text, (corrected, state) = predict_plate(chars, bboxes, segmented_debug)
                                        st.markdown(f'<p class="success">✅ <strong>Recognized Text:</strong> <code>{raw_text}</code></p>', unsafe_allow_html=True)
                                        st.markdown(f'<p class="success">✅ <strong>Corrected Text:</strong> <code>{corrected}</code></p>', unsafe_allow_html=True)
                                        st.markdown(f"🌐 Vehicle registered in: `{state}`")
elif option == "Live Webcam Detection":
    st.subheader("🎥 Live Webcam Detection")
    st.markdown("""
    <div class="card">
        <p>Click the button below to capture an image from your webcam and process it.</p>
    </div>
    """, unsafe_allow_html=True)
    
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    run = st.button("📸 Capture & Process", type="primary")

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera permissions.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Webcam read failed.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            if run:
                with st.spinner("Processing captured image..."):
                    plates, annotated = detect_plates(frame, "webcam")
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Detected Plates", use_column_width=True)
                    
                    if not plates:
                        st.markdown('<p class="warning">⚠️ No license plates detected in the captured image.</p>', unsafe_allow_html=True)
                    else:
                        for idx, (plate_img, box) in enumerate(plates):
                            with st.container():
                                st.markdown(f"### 🚘 Plate {idx + 1} Analysis")
                                cols = st.columns(2)
                                with cols[0]:
                                    st.image(plate_img, caption=f"Plate Region {idx + 1}", use_column_width=True)
                                
                                with cols[1]:
                                    chars, bboxes, segmented_debug = segment_characters(plate_img)
                                    if not chars:
                                        st.markdown('<p class="warning">⚠️ No characters found in this plate.</p>', unsafe_allow_html=True)
                                    else:
                                        raw_text, (corrected, state) = predict_plate(chars, bboxes, segmented_debug)
                                        st.markdown(f'<p class="success">✅ <strong>Recognized Text:</strong> <code>{raw_text}</code></p>', unsafe_allow_html=True)
                                        st.markdown(f"**✅ Corrected Text:** `{corrected}`")
                                        st.markdown(f"🌐 Vehicle registered in: `{state}`")
                break
        cap.release()