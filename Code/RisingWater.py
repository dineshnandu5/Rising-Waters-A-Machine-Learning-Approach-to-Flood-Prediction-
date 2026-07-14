from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
import joblib
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "FloodPredictionSecretKey"

DATABASE = "database/flood.db"
MODEL_PATH = "model/flood_model.pkl"
SCALER_PATH = "model/scaler.pkl"

# -------------------------------------------------------
# Database Connection
# -------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------------------------------------
# Create Tables
# -------------------------------------------------------
def create_tables():
    conn = get_connection()

    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        rainfall REAL,
        temperature REAL,
        humidity REAL,
        water_level REAL,
        soil_moisture REAL,
        prediction TEXT
    )
    """)

    conn.commit()
    conn.close()

create_tables()

# -------------------------------------------------------
# Load ML Model
# -------------------------------------------------------
model = None
scaler = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

# -------------------------------------------------------
# Home
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------------------------------
# Register
# -------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        fullname = request.form["fullname"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        conn = get_connection()

        try:
            conn.execute(
                "INSERT INTO users(fullname,email,password) VALUES(?,?,?)",
                (fullname,email,password)
            )

            conn.commit()
            flash("Registration Successful")

            return redirect(url_for("login"))

        except:

            flash("Email already exists")

        finally:
            conn.close()

    return render_template("register.html")

# -------------------------------------------------------
# Login
# -------------------------------------------------------
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method=="POST":

        email=request.form["email"]
        password=request.form["password"]

        conn=get_connection()

        user=conn.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()

        conn.close()

        if user:

            if check_password_hash(user["password"],password):

                session["user_id"]=user["id"]
                session["fullname"]=user["fullname"]

                return redirect(url_for("dashboard"))

        flash("Invalid Login")

    return render_template("login.html")

# -------------------------------------------------------
# Dashboard
# -------------------------------------------------------
@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn=get_connection()

    total_predictions=conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id=?",
        (session["user_id"],)
    ).fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        name=session["fullname"],
        total=total_predictions
    )

# -------------------------------------------------------
# Predict
# -------------------------------------------------------
@app.route("/predict",methods=["GET","POST"])
def predict():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method=="POST":

        rainfall=float(request.form["rainfall"])
        temperature=float(request.form["temperature"])
        humidity=float(request.form["humidity"])
        water_level=float(request.form["water_level"])
        soil=float(request.form["soil"])

        data=np.array([[rainfall,
                        temperature,
                        humidity,
                        water_level,
                        soil]])

        if scaler:
            data=scaler.transform(data)

        if model:
            result=model.predict(data)[0]
        else:

            if rainfall>200 or water_level>7:
                result="High Risk"
            elif rainfall>120:
                result="Medium Risk"
            else:
                result="Low Risk"

        conn=get_connection()

        conn.execute("""
        INSERT INTO predictions
        (user_id,rainfall,temperature,humidity,water_level,soil_moisture,prediction)
        VALUES(?,?,?,?,?,?,?)
        """,

        (
            session["user_id"],
            rainfall,
            temperature,
            humidity,
            water_level,
            soil,
            str(result)
        )
        )

        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            prediction=result,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity,
            water_level=water_level,
            soil=soil
        )

    return render_template("predict.html")

# -------------------------------------------------------
# History
# -------------------------------------------------------
@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn=get_connection()

    rows=conn.execute("""
    SELECT *
    FROM predictions
    WHERE user_id=?
    ORDER BY id DESC
    """,(session["user_id"],)).fetchall()

    conn.close()

    return render_template(
        "history.html",
        rows=rows
    )

# -------------------------------------------------------
# About
# -------------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")

# -------------------------------------------------------
# Logout
# -------------------------------------------------------
@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))

# -------------------------------------------------------
# Run App
# -------------------------------------------------------
if __name__=="__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )