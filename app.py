"""
app.py - Flask Backend for Crime Rate Prediction System
Run with: python app.py
"""

from flask import Flask, render_template, jsonify, request
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from model import (
    predict, simulate_daily_update, get_city_trends,
    get_all_cities_summary, load_data, train_model, is_festival_day,
    # ── NEW ──
    get_time_of_day_risk, compare_cities, get_safety_scorecard,
)

app = Flask(__name__)


# ─── EXISTING ROUTES (unchanged) ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/cities', methods=['GET'])
def get_cities():
    df = load_data()
    cities = sorted(df['City'].unique().tolist())
    return jsonify({'cities': cities})


@app.route('/api/predict', methods=['POST'])
def predict_city():
    data = request.get_json()
    city = data.get('city', '').strip()
    festival_boost = data.get('festival_boost', True)
    if not city:
        return jsonify({'error': 'City is required'}), 400
    result = predict(city, festival_boost=festival_boost)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route('/api/trends/<city>', methods=['GET'])
def city_trends(city):
    trends = get_city_trends(city)
    if not trends:
        return jsonify({'error': f'No data for city: {city}'}), 404
    return jsonify(trends)


@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    city = data.get('city', '').strip()
    if not city:
        return jsonify({'error': 'City is required'}), 400
    success = simulate_daily_update(city)
    if not success:
        return jsonify({'error': f'Could not simulate data for {city}'}), 400
    return jsonify({'message': f'✅ Simulated new crime data for {city} and retrained model.'})


@app.route('/api/summary', methods=['GET'])
def summary():
    data = get_all_cities_summary()
    return jsonify({'summary': data})


@app.route('/api/festival', methods=['GET'])
def festival_check():
    fest = is_festival_day()
    return jsonify({'festival': fest})


@app.route('/api/retrain', methods=['POST'])
def retrain():
    acc, cities = train_model()
    return jsonify({
        'message': '✅ Model retrained successfully.',
        'accuracy': round(acc * 100, 1),
        'cities_trained': cities,
    })


# ─── NEW ROUTES ───────────────────────────────────────────────────────────────

@app.route('/api/timeofday/<city>', methods=['GET'])
def time_of_day(city):
    """
    Returns hourly risk breakdown for the given city.
    GET /api/timeofday/Mumbai
    """
    result = get_time_of_day_risk(city)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route('/api/compare', methods=['POST'])
def compare():
    """
    Compare 2–5 cities side-by-side.
    POST body: { "cities": ["Mumbai", "Delhi", "Chennai"] }
    """
    data = request.get_json()
    cities = data.get('cities', [])
    if not cities:
        return jsonify({'error': 'cities list is required'}), 400
    result = compare_cities(cities)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/scorecard/<city>', methods=['GET'])
def scorecard(city):
    """
    Return dimension-wise safety scorecard for a city.
    GET /api/scorecard/Delhi
    """
    result = get_safety_scorecard(city)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🚔 Crime Rate Prediction System starting...")
    print("📊 Visit: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
