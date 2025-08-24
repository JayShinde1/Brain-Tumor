#Advanced Brain Tumor Detection and Classification project

<br>
Creator: Jay Shinde


##First clone the project

## Steps to Run the Project

### 1. Train the Model
1. Open and run **tumor.ipynb**.
2. The notebook will generate `model.pkl`.
3. Create a folder named **model** in the project root and place `model.pkl` inside it.

### 2. Set Up the Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv/Scripts/activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


#Run the fastapi backend
uvicorn main:app --reload

#Run the streamlit frontend while running the backend in another terminal.
streamlit run frontend.py
