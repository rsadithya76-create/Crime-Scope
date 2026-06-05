"""
model.py - Crime Rate Prediction ML Model
Uses RandomForestClassifier to predict city safety levels
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import json
from datetime import datetime, timedelta
import random

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'crime_data.xlsx')
SIMULATED_PATH = os.path.join(os.path.dirname(__file__), 'data', 'simulated_updates.json')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'rf_model.pkl')

# Crime weights (higher = more severe)
CRIME_WEIGHTS = {
    'Murder': 10,
    'Kidnapping': 7,
    'Crime against women': 8,
    'Crime against children': 9,
    'Crime Committed by Juveniles': 4,
    'Crime against Senior Citizen': 6,
    'Crime against SC': 5,
    'Crime against ST': 5,
    'Economic Offences': 3,
    'Cyber Crimes': 2,
}

# Festival dates (month-day) that boost crime probability
FESTIVAL_DATES = {
    'Diwali': [(10, 20), (10, 21), (10, 22), (10, 23), (10, 24)],
    'New Year': [(12, 31), (1, 1)],
    'Holi': [(3, 24), (3, 25)],
    'Eid': [(4, 10), (4, 11)],
    'Christmas': [(12, 24), (12, 25)],
}

# ─── TIME OF DAY RISK CONFIG ─────────────────────────────────────────────────
# Multipliers by hour band (based on criminology research patterns)
TIME_RISK_PROFILE = {
    'Early Morning (12AM–5AM)': {'hours': range(0, 5),   'multiplier': 1.6, 'icon': '🌑', 'tip': 'Highest risk window. Avoid isolated areas.'},
    'Morning (5AM–10AM)':       {'hours': range(5, 10),  'multiplier': 0.5, 'icon': '🌅', 'tip': 'Relatively safe. Normal commute hours.'},
    'Daytime (10AM–4PM)':       {'hours': range(10, 16), 'multiplier': 0.4, 'icon': '☀️', 'tip': 'Safest period. High public activity.'},
    'Evening (4PM–8PM)':        {'hours': range(16, 20), 'multiplier': 0.9, 'icon': '🌆', 'tip': 'Moderate risk. Stay in well-lit areas.'},
    'Night (8PM–12AM)':         {'hours': range(20, 24), 'multiplier': 1.3, 'icon': '🌃', 'tip': 'Elevated risk. Travel in groups.'},
}


def load_data():
    """Load and preprocess crime data from Excel file."""
    df = pd.read_excel(DATA_PATH)

    # Rename population column for ease
    df.rename(columns={'Population (in Lakhs) (2011)+': 'Population'}, inplace=True)

    # Compute weighted crime score per row (per capita)
    crime_cols = list(CRIME_WEIGHTS.keys())
    df['raw_score'] = sum(df[col] * w for col, w in CRIME_WEIGHTS.items() if col in df.columns)

    # Normalize by population (crimes per lakh)
    df['crime_per_lakh'] = df['raw_score'] / df['Population'].replace(0, np.nan)
    df['crime_per_lakh'] = df['crime_per_lakh'].fillna(df['crime_per_lakh'].median())

    # Apply simulated daily updates if any exist
    df = apply_simulated_updates(df)

    return df


def apply_simulated_updates(df):
    """Merge any simulated daily crime entries into the dataset."""
    if not os.path.exists(SIMULATED_PATH):
        return df
    try:
        with open(SIMULATED_PATH, 'r') as f:
            updates = json.load(f)
        if updates:
            sim_df = pd.DataFrame(updates)
            # Align columns
            for col in df.columns:
                if col not in sim_df.columns:
                    sim_df[col] = 0
            sim_df = sim_df[df.columns]
            df = pd.concat([df, sim_df], ignore_index=True)
    except Exception:
        pass
    return df


def assign_safety_label(crime_per_lakh, percentiles):
    """Assign SAFE / MODERATE / UNSAFE based on distribution percentiles."""
    low, high = percentiles
    if crime_per_lakh <= low:
        return 'SAFE'
    elif crime_per_lakh <= high:
        return 'MODERATE'
    else:
        return 'UNSAFE'


def train_model():
    """Train RandomForestClassifier and save to disk."""
    df = load_data()

    # Compute percentile thresholds for labelling
    p33 = df['crime_per_lakh'].quantile(0.33)
    p66 = df['crime_per_lakh'].quantile(0.66)

    df['label'] = df['crime_per_lakh'].apply(lambda x: assign_safety_label(x, (p33, p66)))

    # Encode city names
    le = LabelEncoder()
    df['city_enc'] = le.fit_transform(df['City'])

    feature_cols = ['city_enc', 'Year', 'Population',
                    'Murder', 'Kidnapping', 'Crime against women',
                    'Crime against children', 'Crime Committed by Juveniles',
                    'Crime against Senior Citizen', 'Crime against SC',
                    'Crime against ST', 'Economic Offences', 'Cyber Crimes']

    X = df[feature_cols].fillna(0)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))

    # Save model + encoder + thresholds
    payload = {
        'model': clf,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'p33': p33,
        'p66': p66,
        'accuracy': acc,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)

    return acc, le.classes_.tolist()


def load_model():
    """Load trained model from disk (trains fresh if not found)."""
    if not os.path.exists(MODEL_PATH):
        train_model()
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def is_festival_day(dt=None):
    """Return festival name if today is a festival date, else None."""
    if dt is None:
        dt = datetime.now()
    for festival, dates in FESTIVAL_DATES.items():
        if (dt.month, dt.day) in dates:
            return festival
    return None


def compute_crime_score(crime_per_lakh, p33, p66):
    """Return a 0-100 crime score and risk tier."""
    score = min(100, int((crime_per_lakh / (p66 * 1.5)) * 100))
    if score <= 33:
        tier = 'Low'
    elif score <= 66:
        tier = 'Medium'
    else:
        tier = 'High'
    return score, tier


def predict(city, year=None, festival_boost=True):
    """
    Predict safety level for a city.
    Returns dict with label, probability, crime_score, tier, festival_note.
    """
    payload = load_model()
    clf = payload['model']
    le = payload['label_encoder']
    feature_cols = payload['feature_cols']
    p33 = payload['p33']
    p66 = payload['p66']

    df = load_data()
    if year is None:
        year = df['Year'].max()

    city_df = df[df['City'] == city]
    if city_df.empty:
        return {'error': f'City "{city}" not found in dataset.'}

    latest = city_df.sort_values('Year').iloc[-1]

    if city not in le.classes_:
        return {'error': f'City "{city}" not in trained labels.'}
    city_enc = le.transform([city])[0]

    row = {
        'city_enc': city_enc,
        'Year': year,
        'Population': latest.get('Population', 0),
        'Murder': latest.get('Murder', 0),
        'Kidnapping': latest.get('Kidnapping', 0),
        'Crime against women': latest.get('Crime against women', 0),
        'Crime against children': latest.get('Crime against children', 0),
        'Crime Committed by Juveniles': latest.get('Crime Committed by Juveniles', 0),
        'Crime against Senior Citizen': latest.get('Crime against Senior Citizen', 0),
        'Crime against SC': latest.get('Crime against SC', 0),
        'Crime against ST': latest.get('Crime against ST', 0),
        'Economic Offences': latest.get('Economic Offences', 0),
        'Cyber Crimes': latest.get('Cyber Crimes', 0),
    }

    X = pd.DataFrame([row])[feature_cols]
    label = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_

    proba_dict = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    crime_per_lakh = latest['crime_per_lakh']
    score, tier = compute_crime_score(crime_per_lakh, p33, p66)

    festival = is_festival_day() if festival_boost else None
    festival_note = None
    if festival:
        festival_note = f"⚠️ Festival Alert: {festival} – heightened vigilance advised."
        if label == 'SAFE':
            label = 'MODERATE'
        proba_dict['UNSAFE'] = min(100, proba_dict.get('UNSAFE', 0) + 15)
        score = min(100, score + 10)

    return {
        'city': city,
        'year': int(year),
        'label': label,
        'probabilities': proba_dict,
        'crime_score': score,
        'risk_tier': tier,
        'crime_per_lakh': round(float(crime_per_lakh), 2),
        'festival_note': festival_note,
        'model_accuracy': round(payload['accuracy'] * 100, 1),
    }


def simulate_daily_update(city):
    """Simulate a new crime data entry for today for a given city."""
    df = load_data()
    city_df = df[df['City'] == city]
    if city_df.empty:
        return False

    latest = city_df.sort_values('Year').iloc[-1].to_dict()
    latest['Year'] = datetime.now().year

    crime_cols = list(CRIME_WEIGHTS.keys())
    for col in crime_cols:
        if col in latest:
            latest[col] = max(0, int(latest[col] * random.uniform(0.8, 1.2)))

    raw = sum(latest.get(c, 0) * w for c, w in CRIME_WEIGHTS.items())
    pop = latest.get('Population', 1)
    latest['crime_per_lakh'] = raw / pop if pop else 0

    updates = []
    if os.path.exists(SIMULATED_PATH):
        with open(SIMULATED_PATH, 'r') as f:
            updates = json.load(f)

    updates.append(latest)

    with open(SIMULATED_PATH, 'w') as f:
        json.dump(updates, f, default=str)

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    train_model()
    return True


def get_city_trends(city):
    """Return year-wise crime trend data for chart rendering."""
    df = load_data()
    city_df = df[df['City'] == city].sort_values('Year')
    if city_df.empty:
        return {}

    result = {
        'years': city_df['Year'].tolist(),
        'crime_per_lakh': city_df['crime_per_lakh'].round(2).tolist(),
        'murder': city_df['Murder'].tolist(),
        'kidnapping': city_df['Kidnapping'].tolist(),
        'crime_women': city_df['Crime against women'].tolist(),
        'cyber': city_df['Cyber Crimes'].tolist(),
    }
    return result


def get_all_cities_summary():
    """Return latest safety summary for all cities."""
    df = load_data()
    payload = load_model()
    p33 = payload['p33']
    p66 = payload['p66']

    summaries = []
    for city in sorted(df['City'].unique()):
        city_df = df[df['City'] == city].sort_values('Year')
        latest = city_df.iloc[-1]
        cpl = latest['crime_per_lakh']
        score, tier = compute_crime_score(cpl, p33, p66)
        summaries.append({
            'city': city,
            'crime_per_lakh': round(float(cpl), 2),
            'crime_score': score,
            'risk_tier': tier,
        })

    return summaries


# ─── NEW FEATURE 1: TIME OF DAY RISK ─────────────────────────────────────────

def get_time_of_day_risk(city):
    """
    Returns risk multipliers for each time window of the day for a given city.
    Combines base city crime score with time-of-day multipliers.
    """
    result = predict(city, festival_boost=False)
    if 'error' in result:
        return result

    base_score = result['crime_score']
    current_hour = datetime.now().hour

    time_bands = []
    for band_name, info in TIME_RISK_PROFILE.items():
        adjusted = min(100, int(base_score * info['multiplier']))
        is_now = current_hour in info['hours']

        if adjusted <= 33:
            level = 'Low'
            color = 'safe'
        elif adjusted <= 66:
            level = 'Moderate'
            color = 'moderate'
        else:
            level = 'High'
            color = 'unsafe'

        time_bands.append({
            'band': band_name,
            'icon': info['icon'],
            'risk_score': adjusted,
            'level': level,
            'color': color,
            'tip': info['tip'],
            'is_current': is_now,
        })

    # Current window
    current_band = next((b for b in time_bands if b['is_current']), time_bands[0])

    return {
        'city': city,
        'base_score': base_score,
        'current_hour': current_hour,
        'current_band': current_band['band'],
        'current_risk': current_band['risk_score'],
        'current_level': current_band['level'],
        'time_bands': time_bands,
    }


# ─── NEW FEATURE 2: CITY COMPARISON ──────────────────────────────────────────

def compare_cities(city_list):
    """
    Compare multiple cities side-by-side.
    Returns per-city prediction + breakdown data.
    """
    if not city_list or len(city_list) < 2:
        return {'error': 'Provide at least 2 cities to compare.'}
    if len(city_list) > 5:
        return {'error': 'Maximum 5 cities allowed for comparison.'}

    df = load_data()
    payload = load_model()
    p33 = payload['p33']
    p66 = payload['p66']

    results = []
    for city in city_list:
        pred = predict(city, festival_boost=False)
        if 'error' in pred:
            results.append({'city': city, 'error': pred['error']})
            continue

        city_df = df[df['City'] == city].sort_values('Year')
        latest = city_df.iloc[-1]

        breakdown = {
            crime: int(latest.get(crime, 0))
            for crime in CRIME_WEIGHTS.keys()
        }

        # Trend direction: compare last 2 years
        if len(city_df) >= 2:
            prev = city_df.iloc[-2]['crime_per_lakh']
            curr = city_df.iloc[-1]['crime_per_lakh']
            trend = 'Rising ↑' if curr > prev * 1.02 else ('Falling ↓' if curr < prev * 0.98 else 'Stable →')
        else:
            trend = 'N/A'

        results.append({
            'city': city,
            'label': pred['label'],
            'crime_score': pred['crime_score'],
            'risk_tier': pred['risk_tier'],
            'crime_per_lakh': pred['crime_per_lakh'],
            'probabilities': pred['probabilities'],
            'breakdown': breakdown,
            'trend': trend,
        })

    # Rank by crime_score ascending (safest first)
    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda x: x['crime_score'])
    for i, r in enumerate(valid):
        r['rank'] = i + 1

    return {'comparison': results}


# ─── NEW FEATURE 3: SAFETY SCORECARD ─────────────────────────────────────────

SCORECARD_DIMENSIONS = {
    'Violent Crime':   ['Murder', 'Kidnapping'],
    'Gender Safety':   ['Crime against women', 'Crime against children'],
    'Cyber Safety':    ['Cyber Crimes', 'Economic Offences'],
    'Public Order':    ['Crime Committed by Juveniles', 'Crime against Senior Citizen'],
    'Social Equity':   ['Crime against SC', 'Crime against ST'],
}

def get_safety_scorecard(city):
    """
    Generate a detailed safety scorecard for a city with dimension-wise grades.
    """
    df = load_data()
    payload = load_model()
    p33 = payload['p33']
    p66 = payload['p66']

    city_df = df[df['City'] == city].sort_values('Year')
    if city_df.empty:
        return {'error': f'City "{city}" not found.'}

    latest = city_df.iloc[-1]

    # For grading, compare each crime column against all cities
    all_latest = df.groupby('City').last().reset_index()

    dimension_scores = []
    for dim_name, cols in SCORECARD_DIMENSIONS.items():
        dim_vals = []
        for col in cols:
            if col not in all_latest.columns:
                continue
            city_val = float(latest.get(col, 0))
            col_max = float(all_latest[col].max()) or 1
            # Lower crime = better score (invert)
            norm = 1 - (city_val / col_max)
            dim_vals.append(norm)

        if not dim_vals:
            continue

        avg_norm = sum(dim_vals) / len(dim_vals)
        score_100 = round(avg_norm * 100)

        if score_100 >= 75:
            grade, color = 'A', 'safe'
        elif score_100 >= 55:
            grade, color = 'B', 'moderate-safe'
        elif score_100 >= 35:
            grade, color = 'C', 'moderate'
        elif score_100 >= 20:
            grade, color = 'D', 'moderate-unsafe'
        else:
            grade, color = 'F', 'unsafe'

        dimension_scores.append({
            'dimension': dim_name,
            'score': score_100,
            'grade': grade,
            'color': color,
        })

    overall_score = round(sum(d['score'] for d in dimension_scores) / len(dimension_scores)) if dimension_scores else 0

    if overall_score >= 70:
        overall_grade, verdict = 'A', 'Generally Safe City'
    elif overall_score >= 50:
        overall_grade, verdict = 'B', 'Moderate Safety'
    elif overall_score >= 30:
        overall_grade, verdict = 'C', 'Exercise Caution'
    else:
        overall_grade, verdict = 'D', 'High Risk Area'

    # Year-over-year improvement
    yoy = None
    if len(city_df) >= 2:
        prev_score = city_df.iloc[-2]['crime_per_lakh']
        curr_score = city_df.iloc[-1]['crime_per_lakh']
        change_pct = round(((curr_score - prev_score) / prev_score) * 100, 1) if prev_score else 0
        yoy = {'change_pct': change_pct, 'direction': 'Improved' if change_pct < 0 else 'Worsened'}

    return {
        'city': city,
        'overall_score': overall_score,
        'overall_grade': overall_grade,
        'verdict': verdict,
        'dimensions': dimension_scores,
        'year_over_year': yoy,
        'data_year': int(latest.get('Year', 0)),
    }


# Pre-train on startup if model missing
if not os.path.exists(MODEL_PATH):
    train_model()
