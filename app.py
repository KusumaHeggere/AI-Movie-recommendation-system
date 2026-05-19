from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import requests
import sqlite3
import os
import random
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import generate_password_hash, check_password_hash
import joblib

app = Flask(__name__)
CORS(app)

TMDB_API_KEY = "011043f187c2ab0406a80be98c403b40"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Gmail SMTP Configuration for Password Resets
GMAIL_ADDRESS = "YOUR_GMAIL_ADDRESS@gmail.com"
GMAIL_APP_PASSWORD = "YOUR_APP_PASSWORD"

# ML Models
sentiment_model = None
tfidf_vectorizer = None

def load_ml_models():
    global sentiment_model, tfidf_vectorizer
    try:
        if os.path.exists('sentiment_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
            sentiment_model = joblib.load('sentiment_model.pkl')
            tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
            print("Successfully loaded trained ML Sentiment Analysis models!")
    except Exception as e:
        print("ML Model Load Error:", e)

# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            reset_code TEXT,
            reset_expiry REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            movie_id TEXT NOT NULL,
            movie_title TEXT NOT NULL,
            poster_url TEXT,
            rating REAL,
            sentiment_score REAL,
            review_text TEXT,
            sentiment_label TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize
init_db()
load_ml_models()

def fetch_from_tmdb(endpoint, params=None):
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        return None
    if params is None:
        params = {}
    params['api_key'] = TMDB_API_KEY
    
    for attempt in range(3):
        try:
            response = requests.get(f"{TMDB_BASE_URL}{endpoint}", params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"TMDB Fetch Attempt {attempt+1} failed: {e}")
            time.sleep(1) # wait 1s before retry
            
    return None

def format_tmdb_movie(m):
    poster = f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get('poster_path') else "https://images.unsplash.com/photo-1542204165-65bf26472b9b?auto=format&fit=crop&q=80&w=500"
    year = m.get('release_date', '2020')[:4] if m.get('release_date') else 'N/A'
    return {
        "id": m.get('id'),
        "title": m.get('title', 'Unknown'),
        "genre": "Movie",
        "year": year,
        "poster_url": poster,
        "synopsis": m.get('overview', ''),
        "rating": m.get('vote_average', 0.0)
    }

@app.route("/")
def index():
    return render_template("index.html")

# --- AUTH ENDPOINTS ---
@app.route("/api/user/register", methods=["POST"])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not email or not password:
        return jsonify({"error": "Missing fields"}), 400
        
    hashed = generate_password_hash(password)
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed))
        conn.commit()
        conn.close()
        # Smooth auto-login
        return jsonify({"message": "User registered and logged in successfully", "username": username})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or Email already exists"}), 400

@app.route("/api/user/login", methods=["POST"])
def login():
    data = request.json
    username_or_email = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, password, username FROM users WHERE username = ? OR email = ?", (username_or_email, username_or_email))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user[1], password):
        return jsonify({"message": "Login successful", "username": user[2]})
    return jsonify({"error": "Invalid credentials"}), 401

def send_reset_email(to_email, code):
    if GMAIL_ADDRESS == "YOUR_GMAIL_ADDRESS@gmail.com":
        print(f"\n[SIMULATED EMAIL] To: {to_email} | Reset Code: {code}\n")
        return True
        
    msg = MIMEMultipart()
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = "CineMatch Password Reset Code"
    body = f"Your CineMatch password reset code is: {code}\nIt will expire in 10 minutes."
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        text = msg.as_string()
        server.sendmail(GMAIL_ADDRESS, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print("Email Send Error:", e)
        return False

@app.route("/api/user/forgot_password", methods=["POST"])
def forgot_password():
    email = request.json.get('email')
    if not email:
        return jsonify({"error": "Email required"}), 400
        
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if user:
        code = str(random.randint(100000, 999999))
        expiry = time.time() + 600 # 10 minutes
        cursor.execute("UPDATE users SET reset_code = ?, reset_expiry = ? WHERE id = ?", (code, expiry, user[0]))
        conn.commit()
        
        if send_reset_email(email, code):
            conn.close()
            return jsonify({"message": "Reset code sent to email"})
        else:
            conn.close()
            return jsonify({"error": "Failed to send email. Check SMTP setup."}), 500
            
    conn.close()
    # Always return success to prevent email enumeration
    return jsonify({"message": "If that email is registered, a code was sent."})

@app.route("/api/user/reset_password", methods=["POST"])
def reset_password():
    data = request.json
    email = data.get('email')
    code = data.get('code')
    new_password = data.get('new_password')
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, reset_code, reset_expiry FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if user and user[1] == code:
        if time.time() > user[2]:
            conn.close()
            return jsonify({"error": "Reset code expired"}), 400
            
        hashed = generate_password_hash(new_password)
        cursor.execute("UPDATE users SET password = ?, reset_code = NULL, reset_expiry = NULL WHERE id = ?", (hashed, user[0]))
        conn.commit()
        conn.close()
        return jsonify({"message": "Password reset successful"})
        
    conn.close()
    return jsonify({"error": "Invalid reset code"}), 400

# --- MOVIE ENDPOINTS ---
@app.route("/api/search", methods=["GET"])
def search_movies():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify({"results": []})
        
    data = fetch_from_tmdb("/search/movie", {"query": query, "region": "IN"})
    if data and 'results' in data and len(data['results']) > 0:
        results = [format_tmdb_movie(m) for m in data['results'] if m.get('poster_path')][:12]
        return jsonify({"results": results})
            
    return jsonify({"results": []})

def get_ml_recommendations(movie_id, limit=4):
    recommendations = []
    tmdb_data = fetch_from_tmdb(f"/movie/{movie_id}/similar")
    
    if not tmdb_data or not tmdb_data.get('results'):
        tmdb_data = fetch_from_tmdb(f"/movie/{movie_id}/recommendations")
        
    if tmdb_data and 'results' in tmdb_data and len(tmdb_data['results']) > 0:
        candidates = tmdb_data['results'][:20]
        descriptions = [m.get('overview', '') for m in candidates if m.get('overview')]
        
        if len(descriptions) < 2:
            recommendations = [format_tmdb_movie(m) for m in candidates[:limit] if m.get('poster_path')]
        else:
            try:
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(descriptions)
                sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
                sim_scores = sorted(list(enumerate(sim)), key=lambda x: x[1], reverse=True)[1:limit+1]
                for idx, score in sim_scores:
                    recommendations.append(format_tmdb_movie(candidates[idx]))
            except:
                recommendations = [format_tmdb_movie(m) for m in candidates[:limit] if m.get('poster_path')]
                
    if not recommendations:
        fallback = fetch_from_tmdb("/discover/movie", {"with_origin_country": "IN", "sort_by": "popularity.desc"})
        if fallback and 'results' in fallback:
            recommendations = [format_tmdb_movie(m) for m in fallback['results'][:limit] if m.get('poster_path')]
            
    return recommendations

@app.route("/api/recommend", methods=["GET"])
def get_recommendations():
    movie_id = request.args.get("movie_id")
    username = request.args.get("username")
    
    if not movie_id and username:
        # Personalization Engine: Find user's highest rated movie
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT movie_id FROM reviews WHERE username = ? ORDER BY sentiment_score DESC LIMIT 1", (username,))
        row = cursor.fetchone()
        conn.close()
        if row:
            movie_id = row[0]
    
    if not movie_id:
        trending = fetch_from_tmdb("/trending/movie/week")
        indian = fetch_from_tmdb("/discover/movie", {"with_origin_country": "IN", "sort_by": "popularity.desc"})
        
        combined_results = []
        if indian and 'results' in indian:
            combined_results.extend([format_tmdb_movie(m) for m in indian['results'][:8]])
        if trending and 'results' in trending:
            combined_results.extend([format_tmdb_movie(m) for m in trending['results'][:4]])
            
        if combined_results:
            return jsonify({"results": combined_results})
        return jsonify({"results": []})
        
    results = get_ml_recommendations(movie_id, limit=12)
    return jsonify({"results": results})

@app.route("/api/review", methods=["POST"])
def submit_review():
    data = request.json
    review_text = data.get("review_text", "").lower()
    rating = float(data.get("rating", 0)) 
    username = data.get("username")
    movie_id = data.get("movie_id")
    movie_title = data.get("movie_title", "Unknown")
    poster_url = data.get("poster_url", "")
    
    # Use real ML model if loaded
    if sentiment_model is not None and tfidf_vectorizer is not None and review_text:
        X = tfidf_vectorizer.transform([review_text])
        probabilities = sentiment_model.predict_proba(X)[0]
        positive_prob = probabilities[1]
        score = (positive_prob * 0.7) + ((rating / 5) * 0.3)
        
        if score >= 0.6:
            sentiment = "Positive"
            tags = ["Thrilling", "Exciting"] if score > 0.8 else ["Good"]
        elif score <= 0.4:
            sentiment = "Negative"
            tags = ["Disappointing", "Boring"] if score < 0.2 else ["Flawed"]
        else:
            sentiment = "Neutral"
            tags = ["Average", "Mixed Feelings"]
    else:
        # Fallback Mock Logic
        if any(word in review_text for word in ["bad", "terrible", "boring", "awful"]):
            sentiment = "Negative"
            score = max(0.1, rating / 10)
            tags = ["Disappointing"]
        elif any(word in review_text for word in ["good", "great", "amazing", "thrilling", "loved", "fast"]):
            sentiment = "Positive"
            score = min(0.99, (rating / 5) * 0.9)
            tags = ["Thrilling", "Exciting"]
        else:
            sentiment = "Neutral"
            score = 0.5
            tags = ["Average"]
            
    # Save to Database
    if username and movie_id:
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reviews (username, movie_id, movie_title, poster_url, rating, sentiment_score, review_text, sentiment_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, str(movie_id), movie_title, poster_url, rating, score, review_text, sentiment))
            conn.commit()
            conn.close()
        except Exception as e:
            print("DB Review Save Error:", e)
            
    # Immediate Recommendations for the Modal
    suggested_movies = []
    if movie_id:
        suggested_movies = get_ml_recommendations(movie_id, limit=3)
        
    return jsonify({
        "sentiment": sentiment,
        "sentiment_score": score,
        "tone_tags": tags,
        "satisfaction": "Satisfied" if score > 0.6 else "Dissatisfied",
        "recommendations": suggested_movies
    })

@app.route("/api/user/reviews", methods=["GET"])
def get_user_reviews():
    username = request.args.get('username')
    if not username:
        return jsonify({"results": []})
        
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT movie_id, movie_title, poster_url, rating, sentiment_score, review_text, sentiment_label 
            FROM reviews WHERE username = ? ORDER BY id DESC
        ''', (username,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "title": r[1],
                "poster_url": r[2],
                "rating": r[3],
                "sentiment_score": r[4],
                "review_text": r[5],
                "sentiment": r[6]
            })
        return jsonify({"results": results})
    except Exception as e:
        print("Fetch Reviews Error:", e)
        return jsonify({"results": []})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
