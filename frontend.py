import streamlit as st
import requests
from PIL import Image
import io
import json

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Advanced Brain Tumor Classification System")
st.markdown("Upload a brain MRI scan and fill in patient details for AI-powered tumor Classification.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload & Patient Info")
    
    with st.form("prediction_form"):
        image_file = st.file_uploader(
            "Upload Brain MRI Scan", 
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )

        name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name"
        )

        age = st.number_input(
            "Patient Age", 
            min_value=1, 
            max_value=119, 
            value=30,
            step=1
        )
        
        sex = st.selectbox(
            "Patient Sex", 
            ["Male", "Female"]
        )
        
        submit = st.form_submit_button("🔍 Analyze MRI Scan", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    
    if submit:
        if image_file is None:
            st.error("❌ Please upload an MRI image first.")
        else:
            try:
                st.image(
                    Image.open(image_file), 
                    caption="Uploaded MRI Scan", 
                    use_container_width =True
                )
                
                image_file.seek(0)
                
                files = {"file": (image_file.name, image_file, image_file.type)}
                data = {"name": name, "age": str(age), "sex": sex}
                
                with st.spinner("🤖 AI is analyzing the MRI scan..."):
                    response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    prediction = result.get("Prediction", "Unknown")
                    accuracy = result.get("Accuracy", "0%")
                    
                    if prediction.lower() == "notumor":
                        st.success(f"✅ **No Tumor Detected**")
                        st.info(f"🎯 Confidence: {accuracy}")
                    else:
                        st.warning(f"⚠️ **Tumor Detected: {prediction.title()}**")
                        st.info(f"🎯 Confidence: {accuracy}")
                    
                    if "All predictions: " in result:
                        st.subheader("📈 Detailed Analysis")
                        all_preds = result["All predictions"]
                        
                        for tumor_type, confidence in all_preds.items():
                            if tumor_type.lower() == "notumor":
                                label = "No Tumor"
                            else:
                                label = tumor_type.title()
                            
                            st.write(f"**{label}:** {confidence}%")
                            st.progress(confidence / 100)
                    
                    if "Patient info." in result:
                        patient_info = result["Patient info."]
                        st.subheader("👤 Patient Information")
                        st.write(f"**Name:** {patient_info['name']}")
                        st.write(f"**Age:** {patient_info['age']} years")
                        st.write(f"**Sex:** {patient_info['sex']}")

                elif response.status_code == 422:
                    st.error("Invalid input data. Please check age and sex values.")
                elif response.status_code == 404:
                    st.error("Invalid image format. Please upload JPG, PNG, or BMP files.")
                else:
                    st.error(f"Server Error ({response.status_code})")
                    st.text(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the AI server. Please ensure the FastAPI server is running on http://localhost:8000")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI-powered system can detect and classify brain tumors from MRI scans.
    
    **Tumor Types Detected:**
    -  Glioma
    -  Meningioma  
    -  No Tumor
    -  Pituitary
    
    **Instructions:**
    1. Upload a clear MRI brain scan.
    2. Enter Patient's name.
    3. Enter Patient's age.
    4. Select patient sex: Male or Female.
    5. Click 'Analyze MRI Scan.
    """)

st.markdown("---")
st.markdown("🏥 **Brain Tumor Classification System** | Powered by AI & FastAPI")
st.markdown("Thankyou!")