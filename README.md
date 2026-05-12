# 🚗 Car Mileage Prediction App

A machine learning web application built with **Streamlit** that predicts a car's mileage (km/l) based on key vehicle attributes.

---

## 🔗 Live Demo

> Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting this repository.

---

## 📌 Features

- Predicts car mileage using a trained machine learning model
- Simple and interactive web UI powered by Streamlit
- Supports different fuel types, transmission types, and car brands
- Fast inference using pre-trained `scikit-learn` model

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Streamlit | Web application framework |
| scikit-learn | Machine learning model |
| pandas | Data manipulation |
| numpy | Numerical operations |
| joblib | Model serialization |

---

## 📁 Project Structure

```
mileage_predict/
│
├── app.py                              # Streamlit application
├── car_mod.pkl                         # Trained ML model (joblib)
├── car_col.pkl                         # Feature column names (joblib)
├── mileage_prediction_assignment.ipynb # Model training notebook
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/achyutgyawali/mileage_predict.git
cd mileage_predict
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repository to **GitHub**
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New app"** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🚀

---

## 📊 Model Inputs

| Field | Type | Description |
|-------|------|-------------|
| Year | Number | Manufacturing year of the car |
| Kilometers Driven | Number | Total km driven |
| Engine | Number | Engine displacement (CC) |
| Power | Number | Engine power (bhp) |
| Seats | Number | Number of seats |
| Fuel Type | Text | e.g., `Petrol`, `Diesel`, `CNG` |
| Transmission | Text | e.g., `Manual`, `Automatic` |
| Brand | Text | Car brand e.g., `Maruti`, `Honda` |

---

## 📈 Model Details

- **Algorithm**: Trained using scikit-learn (see notebook for details)
- **Target**: Mileage in km/l
- **Preprocessing**: One-hot encoding for categorical variables

---

