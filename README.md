# 🔵 CrimeScope — Crime Rate Prediction System

A **Machine Learning-powered web application** that predicts urban safety levels using historical crime data.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
crime_prediction/
├── app.py                  # Flask backend (all API routes)
├── model.py                # ML training & prediction engine
├── requirements.txt
├── data/
│   ├── crime_data.xlsx     # Historical dataset (19 Indian cities, 2014–2021)
│   ├── rf_model.pkl        # Trained model (auto-generated)
│   └── simulated_updates.json  # Real-time simulated data (auto-generated)
├── templates/
│   └── index.html          # Main UI
└── static/
    ├── style.css           # Dark tactical theme
    └── script.js           # Frontend logic + Chart.js
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Safety Prediction | SAFE / MODERATE / UNSAFE per city |
| 📊 Crime Score | 0–100 weighted score (Low / Medium / High) |
| 📈 Trend Charts | Year-wise crime intensity + category breakdown |
| ⚡ Simulate Daily Data | Adds random fluctuation and retrains model |
| 🎉 Festival Mode | Boosts crime probability on festival dates |
| 🗺️ City Overview | All 19 cities color-coded at a glance |

---

## 🧠 ML Model

- **Algorithm**: `RandomForestClassifier` (200 estimators)
- **Accuracy**: ~87%
- **Features**: City, Year, Population, Murder, Kidnapping, Crime vs Women/Children, Juvenile Crime, Senior Citizen Crime, SC/ST Crime, Economic Offences, Cyber Crimes
- **Output Labels**: `SAFE`, `MODERATE`, `UNSAFE`

### Crime Weighting
| Crime Type | Weight |
|---|---|
| Murder | 10 |
| Crime against children | 9 |
| Crime against women | 8 |
| Kidnapping | 7 |
| Crime against Senior Citizens | 6 |
| SC/ST crimes | 5 |
| Juvenile crime | 4 |
| Economic offences | 3 |
| Cyber crimes | 2 |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/cities` | List all cities |
| POST | `/api/predict` | Predict safety for a city |
| GET | `/api/trends/<city>` | Crime trend data |
| GET | `/api/summary` | All-cities overview |
| POST | `/api/simulate` | Simulate new day + retrain |
| POST | `/api/retrain` | Force model retrain |
| GET | `/api/festival` | Check if today is festival |

---

## 📦 Dataset

- **Source**: `crp.xlsx` — 19 major Indian cities, years 2014–2021
- **Records**: 152 rows × 13 columns
- **Cities**: Ahmedabad, Bengaluru, Chennai, Coimbatore, Delhi, Ghaziabad, Hyderabad, Indore, Jaipur, Kanpur, Kochi, Kolkata, Kozhikode, Lucknow, Mumbai, Nagpur, Patna, Pune, Surat
