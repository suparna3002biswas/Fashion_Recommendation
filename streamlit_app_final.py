import streamlit as st
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import hashlib
import sqlite3
import secrets
from datetime import datetime, timedelta
from difflib import get_close_matches
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

@st.cache_data
def load_color_data():
    df = pd.read_csv("/content/drive/MyDrive/topwear_images.csv")
    return df, sorted(df['baseColour'].dropna().unique().tolist())

def set_custom_css():
    st.markdown("""
    <style>
    /* Main background with peach milk gradient */
    .stApp {
        background-color: #FAACA8;
        background-image: linear-gradient(19deg, #FAACA8 0%, #DDD6F3 100%);
        background-size: cover;
    }
    
    /* Sidebar styling - fashion inspired */
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        padding: 20px !important;
        margin: 15px 10px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Card styling with fashion theme */
    .stContainer, .stExpander, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.98) !important;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        color: #d6336c;
    }
    
    .stContainer:hover, .stExpander:hover, .stFileUploader:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }
    
    /* Fashion-themed buttons */
    .stButton>button, .stFormSubmitButton>button {
        background-color: #ff6b81 !important;
        color: black !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%) !important;
    }

    /* Input fields */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        color: black !important;
    }
    
    /* Headers gradient removal */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    h2 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    h3, h4, h5, h6 {
        font-size: 1.8rem !important;
        font-weight: 500 !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: black !important;
    }

    /* Image styling - fashion showcase */
    .stImage>img, .image-container img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    
    .stImage>img:hover, .image-container img:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        border-radius: 10px;
    }
    
    /* Fashion tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 10px 20px !important;
        transition: all 0.3s !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #ff6b81 !important;
        font-weight: 600 !important;
    }
    
    /* Fashion spinner */
    .stSpinner>div>div {
        border-color: #ff9a9e transparent transparent transparent !important;
    }
    
    /* Image containers with hover effect */
    .image-container {
        width: 100% !important;
        height: 250px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
        margin-bottom: 15px !important;
        border-radius: 12px !important;
        border: 1px solid #e0e0e0 !important;
        cursor: pointer !important;
        transition: all 0.3s !important;
        position: relative;
    }
        
    .image-container:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12) !important;
    }
        
    .image-container img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        transition: transform 0.5s !important;
    }
        
    .image-container:hover img {
        transform: scale(1.1);
    }
        
    /* Image caption styling */
    .image-caption {
        font-size: 0.9rem;
        text-align: center;
        color: #666;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    
    /* Fashion badge for similarity score */
    .similarity-badge {
        position: absolute;
        bottom: 10px;
        right: 10px;
        background: rgba(255, 255, 255, 0.9);
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #ff6b81;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Fashion tooltip */
    .stTooltip {
        background: white !important;
        color: #333 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        border: 1px solid #e0e0e0 !important;
    }

    /* Spinner color */
    .stSpinner>div>div {
        border-color: #ff6b81 transparent transparent transparent !important;
    }

    /* Cursor color */
    input, textarea {
        caret-color: black !important;
    }

    /* Dropdown pointer */
    div[data-baseweb="select"] div {
        cursor: pointer !important;
    }

    /* All other text elements in black */
    p {
        color: black !important;
    }

  @media screen and (max-width: 768px) {
    /* Mobile-specific adjustments */
    .stImage>img, .image-container img {
        width: 100% !important;
        height: auto !important;
    }
    
    .stButton>button {
        width: 100% !important;
    }
    
    .image-container {
        height: 180px !important;
    }
    
    /* Stack columns vertically */
    [data-testid="column"] {
        width: 100% !important;
    }
    
    /* Adjust font sizes */
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
}
    </style>
    """, unsafe_allow_html=True)

def is_valid_email(email):
    import re
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def create_users_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, 
                  password TEXT,
                  name TEXT,
                  age INTEGER,
                  alt_email TEXT,
                  gender TEXT)''')
    
    c.execute("PRAGMA table_info(users)")
    existing_columns = [column[1].lower() for column in c.fetchall()]
    
    if 'style_preference' not in existing_columns:
        c.execute("ALTER TABLE users ADD COLUMN style_preference TEXT")
    if 'favorite_colors' not in existing_columns:
        c.execute("ALTER TABLE users ADD COLUMN favorite_colors TEXT")
    if 'gender' not in existing_columns:
        c.execute("ALTER TABLE users ADD COLUMN gender TEXT")
    
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset
                 (email TEXT NOT NULL,
                  token TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (email, token))''')
    
    conn.commit()
    conn.close()

def add_user(email, password, name=None, alt_email=None, style_pref=None, fav_colors=None, gender=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""INSERT INTO users 
               (email, password, name, alt_email, style_preference, favorite_colors, gender) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""", 
               (email, password, name, alt_email, style_pref, fav_colors, gender))
    conn.commit()
    conn.close()

def update_user_details(email, name=None, age=None, alt_email=None, style_pref=None, fav_colors=None, gender=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    updates = []
    params = []
    
    if alt_email:
        c.execute("SELECT email FROM users WHERE email=?", (alt_email,))
        if c.fetchone():
            conn.close()
            raise ValueError("Alternative email already exists as primary email")
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if age is not None:
        updates.append("age = ?")
        params.append(age)
    if alt_email is not None:
        updates.append("alt_email = ?")
        params.append(alt_email)
    if style_pref is not None:
        updates.append("style_preference = ?")
        params.append(style_pref)
    if fav_colors is not None:
        updates.append("favorite_colors = ?")
        params.append(fav_colors)
    if gender is not None:
        updates.append("gender = ?")
        params.append(gender)
    
    if updates:
        update_query = "UPDATE users SET " + ", ".join(updates) + " WHERE email = ?"
        params.append(email)
        c.execute(update_query, params)
    
    conn.commit()
    conn.close()

def get_user_details(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, age, alt_email, style_preference, favorite_colors, gender FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    
    return {
        'name': result[0] if result else None,
        'age': result[1] if result else None,
        'alt_email': result[2] if result else None,
        'style_preference': result[3] if result else None,
        'favorite_colors': result[4] if result else None,
        'gender': result[5] if result else None
    } if result else None

def check_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("""SELECT * FROM users 
              WHERE (email=? OR alt_email=?) AND password=?""", 
              (email, email, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def user_exists(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    return result is not None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_strong_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    return True, ""

def signup():
    df, colors = load_color_data()
    with st.container():
        st.subheader("👑 Create Your Fashion Account")
        col1, col2 = st.columns(2)
        with col1:
            email = st.text_input("Email")
            name = st.text_input("Name (optional)")
            alt_email = st.text_input("Alternative Email (optional)")
        with col2:
            password = st.text_input("Password", type='password')
            confirm_password = st.text_input("Confirm Password", type='password')
        
        st.markdown("### 🎨 Tell Us About Your Style")
        style_col1, style_col2 = st.columns(2)
        with style_col1:
            style_pref = st.selectbox("Your Style Preference", 
                                    ["Casual", "Formal", "Wedding", "Winter", "Daily", "Festive", "Summer", "Common", "Party"])
            gender = st.selectbox("Gender", ["Not prefer to say", "Men", "Women"])
        with style_col2:
            fav_colors = st.multiselect("Favorite Colors", 
                                      options=colors,
                                      help="Select your preferred clothing colors")
        
        if st.button("✨ Sign Up", key="signup_btn", use_container_width=True):
            if not is_valid_email(email):
                st.error("Please enter a valid email address!")
            elif password != confirm_password:
                st.error("Passwords do not match!")
            else:
                is_strong, msg = is_strong_password(password)
                if not is_strong:
                    st.error(f"Weak password: {msg}")
                elif user_exists(email):
                    st.error("Email already exists!")
                else:
                    if not is_valid_email(alt_email):
                            st.error("Invalid alternative email format!")
                    elif alt_email and user_exists(alt_email):
                        st.error("Alternative email already registered!")
                    hashed_password = hash_password(password)
                    add_user(email=email,
                            password=hashed_password,
                            name=name,
                            alt_email=alt_email,
                            style_pref=style_pref,
                            fav_colors=",".join(fav_colors) if fav_colors else None,
                            gender=gender
                          )
                    st.balloons()
                    st.success("Fashion account created successfully! Please login.")
                    st.session_state['page'] = 'login'
                    st.rerun()

def generate_reset_token(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE email=? OR alt_email=?", (email, email))
    result = c.fetchone()
    if not result:
        return None
    actual_email = result[0]
    
    c.execute("DELETE FROM password_reset WHERE email=?", (actual_email,))
    token = secrets.token_hex(3).upper()
    c.execute("INSERT INTO password_reset (email, token) VALUES (?, ?)", 
             (actual_email, token))
    conn.commit()
    conn.close()
    return token

def verify_reset_token(email, token):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM password_reset 
              WHERE email=? AND token=? 
              AND created_at >= datetime('now', '-15 minutes')''',
              (email, token))
    result = c.fetchone()
    conn.close()
    return result is not None

def update_password(email, new_password):
    hashed = hash_password(new_password)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET password=? WHERE email=?", (hashed, email))
    conn.commit()
    conn.close()

def login():
    with st.container():
        st.subheader("👠 Welcome Back, Fashionista!")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            
            if st.form_submit_button("👢 Login", use_container_width=True):
                hashed_password = hash_password(password)
                if check_user(email, hashed_password):
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute("""SELECT email FROM users 
                              WHERE email=? OR alt_email=? AND password=?""", 
                              (email, email, hashed_password))
                    actual_email = c.fetchone()[0]
                    conn.close()
                    st.session_state['logged_in'] = True
                    st.session_state['email'] = actual_email
                    st.balloons()
                    st.rerun()
                else:
                    if user_exists(email):
                        token = generate_reset_token(email)
                        st.session_state['reset_email'] = email
                        st.session_state['show_reset'] = True
                        st.error("Incorrect password. Reset code sent to your email.")
                        st.info(f"Debug: Your reset code is {token}")
                    else:
                        st.error("Invalid email or password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 Don't have an account? Sign Up", use_container_width=True):
                st.session_state['page'] = 'signup'
                st.rerun()
        with col2:
            if st.button("🔑 Forgot Password?", use_container_width=True):
                st.session_state['forgot_password_flow'] = True
                st.rerun()

        if st.session_state.get('show_reset'):
            with st.expander("🔐 Password Reset", expanded=True):
                with st.form("password_reset_form"):
                    code = st.text_input("Reset Code")
                    new_pass = st.text_input("New Password", type='password')
                    confirm_pass = st.text_input("Confirm Password", type='password')
                    
                    if st.form_submit_button("💫 Reset Password", use_container_width=True):
                        if new_pass != confirm_pass:
                            st.error("Passwords don't match!")
                        elif verify_reset_token(st.session_state['reset_email'], code):
                            update_password(st.session_state['reset_email'], new_pass)
                            st.success("Password updated! Please login.")
                            del st.session_state['show_reset']
                            del st.session_state['reset_email']
                            st.rerun()
                        else:
                            st.error("Invalid or expired code")

def handle_forgot_password():
    with st.container():
        st.subheader("🔍 Reset Your Password")
        
        with st.form("forgot_email_form"):
            email = st.text_input("Enter your registered email")
            
            if st.form_submit_button("📬 Send Reset Code", use_container_width=True):
                if user_exists(email):
                    token = generate_reset_token(email)
                    st.session_state['reset_email'] = email
                    st.session_state['show_reset'] = True
                    st.success("Reset code sent to your email.")
                    st.info(f"Debug: Your reset code is {token}")
                else:
                    st.error("Email not found in our system")

        if st.session_state.get('show_reset'):
            with st.expander("✨ Enter Reset Code", expanded=True):
                with st.form("forgot_reset_form"):
                    code = st.text_input("Enter Reset Code")
                    new_pass = st.text_input("New Password", type='password')
                    confirm_pass = st.text_input("Confirm Password", type='password')
                    
                    if st.form_submit_button("🌟 Reset Password", use_container_width=True):
                        if new_pass != confirm_pass:
                            st.error("Passwords don't match!")
                        elif verify_reset_token(st.session_state['reset_email'], code):
                            update_password(st.session_state['reset_email'], new_pass)
                            st.success("Password updated! Please login.")
                            del st.session_state['show_reset']
                            del st.session_state['reset_email']
                            st.session_state['forgot_password_flow'] = False
                            st.rerun()
                        else:
                            st.error("Invalid or expired code")

        if st.button("↩️ Back to Login", use_container_width=True):
            st.session_state['forgot_password_flow'] = False
            st.rerun()

def user_details_sidebar():
    df, colors = load_color_data()
    with st.sidebar:
        user_details = get_user_details(st.session_state['email'])
        
        if user_details is None:
            user_details = {'name': None, 'age': None, 'alt_email': None, 'style_preference': None, 'favorite_colors': None, 'gender': None}
        
        display_name = user_details['name'] or st.session_state['email'].split('@')[0]
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin-bottom: 5px;">👋 {display_name}</h2>
            <p style="color: #666; margin-top: 0;">Your Fashion Profile</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🛍️ Style Preferences", expanded=False):
            with st.form("user_details_form"):
                name = st.text_input("Name", value=user_details['name'] or "")
                age = st.number_input("Age", min_value=1, max_value=120, 
                                    value=user_details['age'] or 18)
                gender = st.selectbox(
                    "Gender",
                    options=["Not prefer to say", "Men", "Women"],
                    index=["Not prefer to say", "Men", "Women"].index(
                        user_details['gender'] or "Not prefer to say")
                )
                alt_email = st.text_input("Alternative Email", 
                                        value=user_details['alt_email'] or "")
                
                style_pref = st.selectbox(
                    "Your Style", 
                    ["Casual", "Formal", "Wedding", "Winter", "Daily", "Festive", "Summer", "Common", "Party"],
                    index=["Casual", "Formal", "Wedding", "Winter", "Daily", "Festive", "Summer", "Common", "Party"].index(
                        user_details['style_preference'] or "Casual")
                )
                
                fav_colors = st.multiselect(
                    "Favorite Colors",
                    options=colors,
                    default=user_details['favorite_colors'].split(",") if user_details['favorite_colors'] else []
                )
                
                if st.form_submit_button("💾 Update Profile", use_container_width=True):
                    update_user_details(
                        email=st.session_state['email'],
                        name=name if name else None,
                        age=age if age else None,
                        gender=gender,
                        alt_email=alt_email if alt_email else None,
                        style_pref=style_pref,
                        fav_colors=",".join(fav_colors) if fav_colors else None
                    )
                    st.success("Profile updated successfully!")
                    st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['email'] = None
            st.rerun()
      
        quotes = [
            "Style is a way to say who you are without having to speak.",
            "Fashion is the armor to survive the reality of everyday life.",
            "Clothes mean nothing until someone lives in them.",
            "Fashion is about dressing according to what's fashionable. Style is more about being yourself.",
            "Elegance is the only beauty that never fades."
        ]
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px; margin-top: 20px;">
            <p style="font-style: italic; color: #555;">💬 Fashion Quote of the Day:</p>
            <p style="font-weight: 500;">"{random.choice(quotes)}"</p>
        </div>
        """, unsafe_allow_html=True)

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def recommendation_system():
    df, colors = load_color_data()
    st.title("👗 Fashion Recommendation System")
    
    user_details = get_user_details(st.session_state['email'])
    if user_details and user_details['style_preference']:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); 
                    padding: 15px; border-radius: 12px; margin-bottom: 20px;
                    border-left: 5px solid #ff9a9e;">
            <h3 style="margin: 0;">✨ Hello, {user_details['name'] or 'Fashionista'}!</h3>
            <p style="margin: 5px 0 0 0;">We've curated recommendations based on your <b>{user_details['style_preference']}</b> style.</p>
        </div>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_models():
        try:
            model1 = load_model('/content/Models/personalized_fashion_densenet201.keras', compile=False)
            model2 = load_model('/content/Models/personalized_fashion_mobilenet_v2.keras', compile=False)
            model3 = load_model('/content/Models/personalized_fashion_Xception.keras', compile=False)
            feature_extractor = Model(inputs=model1.input, outputs=model1.layers[-3].output)
            return [model1, model2, model3], feature_extractor
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None, None

    models, feature_extractor = load_models()
    if models is None or feature_extractor is None:
        return

    dataset_path = "/content/Fashion_Recommendation/Fashion_Reccomendation1"
    
    @st.cache_resource
    def get_label_encoder(dataset_path):
        try:
            labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            return label_encoder
        except Exception as e:
            st.error(f"Error creating label encoder: {str(e)}")
            return None
    
    label_encoder = get_label_encoder(dataset_path)
    if label_encoder is None:
        return

    all_categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    tab1, tab2, tab3 = st.tabs(["🔍 Upload Image", "🎉 Find by Occasion", "🌟 Personalized Picks"])
    
    with tab1:
        st.subheader("📸 Upload Your Fashion Item")
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader("Choose an image", 
                                             type=['png', 'jpg', 'jpeg', 'webp', 'jfif'],
                                             key="uploader_tab1",
                                             help="Upload a clear photo of your clothing item for best results")
            with col2:
                st.markdown("""
                <div style="background: rgba(255,154,158,0.1); padding: 10px; border-radius: 10px;">
                    <p style="font-size: 1.0rem; margin: 0;">💡 <b>Tips:</b></p>
                    <ul style="font-size: 0.9rem; margin: 5px 0 0 15px; padding: 0; color: black;">
                        <li>Use well-lit photos</li>
                        <li>Show the full item</li>
                        <li>Avoid busy backgrounds</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            uploaded_path = os.path.join(upload_dir, uploaded_file.name)
            with open(uploaded_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.subheader("🖼️ Your Uploaded Item")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_file, width=250)
            with col2:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px;">
                    <h4 style="margin-top: 0;">Analyzing your fashion item...</h4>
                    <p style="margin-bottom: 5px;">We're examining the style, color, and patterns of your uploaded item.</p>
                    <p style="margin: 0;">This might take a few moments ⏳</p>
                </div>
                """, unsafe_allow_html=True)
            
            img = image.load_img(uploaded_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            
            with st.spinner('🔍 Analyzing your fashion style...'):
                all_preds = [model.predict(img_preprocessed)[0] for model in models]
                avg_pred = np.mean(all_preds, axis=0)
                predicted_class_index = np.argmax(avg_pred)
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); 
                        padding: 15px; border-radius: 12px; margin: 20px 0;
                        border-left: 5px solid #ff9a9e;">
                <h3 style="margin: 0 0 5px 0;">🏷️ Our Analysis</h3>
                <p style="margin: 0; font-size: 1.2rem;">This looks like <b>{predicted_class}</b> style</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner('✨ Finding similar items from our collection...'):
                predicted_class_dir = os.path.join(dataset_path, predicted_class)
                predicted_class_images = [os.path.join(predicted_class_dir, f)
                                      for f in os.listdir(predicted_class_dir)
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', 'jfif'))
                                      and not f.startswith('aug')]

                query_feature = feature_extractor.predict(img_preprocessed)
                similar_images = []
                
                for img_path in predicted_class_images:
                    try:
                        img_cand = image.load_img(img_path, target_size=(224, 224))
                        img_cand_array = image.img_to_array(img_cand)
                        img_cand_batch = np.expand_dims(img_cand_array, axis=0)
                        img_cand_preprocessed = preprocess_input(img_cand_batch)

                        cand_feature = feature_extractor.predict(img_cand_preprocessed)
                        sim_score = cosine_similarity(query_feature, cand_feature)[0][0]

                        if sim_score >= 0.7 and img_path != uploaded_path:
                            similar_images.append((img_path, sim_score))
                    except Exception as e:
                        continue

                similar_images.sort(key=lambda x: x[1], reverse=True)
                top_similar = similar_images[:10]

            st.subheader(f"💎 Recommended Similar Items")
            
            if len(top_similar) > 0:
                st.markdown(f"""
                <div style="background: rgba(255,154,158,0.1); padding: 10px; border-radius: 10px; margin-bottom: 20px;">
                    <p style="margin: 0;">We found <b>{len(top_similar)}</b> items that match your style with high similarity</p>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(4)
                for idx, (path, sim) in enumerate(top_similar):
                    with cols[idx % 4]:
                        st.markdown(
                            f"""
                            <div class="image-container">
                                <img src="data:image/png;base64,{image_to_base64(path)}">
                                <div class="similarity-badge">{sim:.2f}</div>
                            </div>
                            <div class="image-caption">{os.path.basename(path).split('.')[0]}</div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.warning("No similar items found above similarity threshold. Try uploading a different image.")

    with tab2:
        st.subheader("🎯 Find Clothes by Occasion")
        occasion_options = {
            "Formal": "👔",
            "Casual": "👕", 
            "Party": "🎉",
            "Wedding": "💍",
            "Daily": "💼",
            "Winter": "❄️",
            "Summer": "☀️",
            "Festive": "🎊"
        }
        
        selected_occasion = st.selectbox(
            "Select an occasion:", 
            list(occasion_options.keys()),
            format_func=lambda x: f"{occasion_options[x]} {x}",
            key="occasion_select",
            help="Choose the occasion you're dressing for"
        )
        
        if st.button("Find Recommendations", key="occasion_btn", use_container_width=True):
            with st.spinner(f'🔍 Finding perfect {selected_occasion} outfits...'):
                matching_categories = [
                    cat for cat in all_categories 
                    if selected_occasion.lower() in cat.lower()
                ]
                
                if not matching_categories:
                    matches = get_close_matches(
                        selected_occasion.lower(), 
                        [cat.lower() for cat in all_categories], 
                        n=1, 
                        cutoff=0.4
                    )
                    if matches:
                        matching_categories = [all_categories[[cat.lower() for cat in all_categories].index(matches[0])]]
                
                if matching_categories:
                    matched_category = matching_categories[0]
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); 
                                padding: 15px; border-radius: 12px; margin: 20px 0;
                                border-left: 5px solid #667eea;">
                        <h3 style="margin: 0 0 5px 0;">{occasion_options[selected_occasion]} Perfect for {selected_occasion}</h3>
                        <p style="margin: 0;">We found items from our <b>{matched_category}</b> collection</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    category_dir = os.path.join(dataset_path, matched_category)
                    category_images = [
                        os.path.join(category_dir, f)
                        for f in os.listdir(category_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', 'jfif'))
                        and not f.startswith('aug')
                    ]
                    
                    if category_images:
                        display_images = category_images[:min(12, len(category_images))]
                        cols = st.columns(4)
                        for idx, img_path in enumerate(display_images):
                            with cols[idx % 4]:
                                st.markdown(
                                    f"""
                                    <div class="image-container">
                                        <img src="data:image/png;base64,{image_to_base64(img_path)}">
                                    </div>
                                    <div class="image-caption">{os.path.basename(img_path).split('.')[0]}</div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.warning(f"No images found in the {matched_category} category")
                else:
                    st.error(f"Couldn't find a matching category for '{selected_occasion}'")
                    st.info("Available categories: " + ", ".join(all_categories[:10]) + "...")

    with tab3:
        st.subheader("🌟 Personalized Recommendations")
        
        if not user_details or not user_details['style_preference']:
            st.warning("Complete your style profile in the sidebar to get personalized recommendations!")
        else:
            with st.spinner(f'✨ Finding {user_details["style_preference"]} style items you might love...'):
                preferred_categories = [
                    cat for cat in all_categories 
                    if user_details['style_preference'].lower() in cat.lower()
                ]
                
                if not preferred_categories:
                    preferred_categories = random.sample(all_categories, min(3, len(all_categories)))
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); 
                            padding: 15px; border-radius: 12px; margin: 20px 0;
                            border-left: 5px solid #764ba2;">
                    <h3 style="margin: 0 0 5px 0;">✨ Curated for Your {user_details['style_preference']} Style</h3>
                    <p style="margin: 0;">Based on your profile preferences</p>
                </div>
                """, unsafe_allow_html=True)
                
                for category in preferred_categories:
                      category_dir = os.path.join(dataset_path, category)
                category_images = [
                    os.path.join(category_dir, f)
                    for f in os.listdir(category_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', 'jfif'))
                    and not f.startswith('aug')
                ]
                
                if category_images:
                    st.markdown(f"### {category}")
                    display_images = random.sample(category_images, min(4, len(category_images)))
                    cols = st.columns(4)
                    for idx, img_path in enumerate(display_images):
                        with cols[idx % 4]:
                            st.markdown(
                                f"""
                                <div class="image-container">
                                    <img src="data:image/png;base64,{image_to_base64(img_path)}">
                                </div>
                                <div class="image-caption">{os.path.basename(img_path).split('.')[0]}</div>
                                """, 
                                unsafe_allow_html=True
                            )
            
            if user_details.get('favorite_colors'):
                fav_colors = user_details['favorite_colors'].split(',')
                st.markdown("---")
                st.subheader(f"🎨 Recommendations in Your Favorite Colors")
                
                for color in fav_colors:
                    filtered_df = df[df['baseColour'] == color].head(8)
                    if not filtered_df.empty:
                        image_data = list(zip(filtered_df['link'], filtered_df['filename']))
                        
                        st.markdown(f"### {color.capitalize()} Collection")
                        
                        cols = st.columns(4)
                        
                        for idx in range(8):
                            with cols[idx%4]:
                                if idx < len(image_data):
                                    try:
                                        url, filename = image_data[idx]
                                        st.markdown(
                                            f"""
                                            <div class="image-container">
                                                <img src="{url}">
                                            </div>
                                            <div class="image-caption">{filename.split('.')[0]}</div>
                                            """, 
                                            unsafe_allow_html=True
                                        )
                                    except Exception as e:
                                        st.error(f"Couldn't load image: {filename}")
                                else:
                                    st.empty()
                        
                        st.markdown("---")
                    else:
                        st.warning(f"We couldn't find items matching your favourite color {color} preferences")
def main():
    set_custom_css()
    create_users_db()
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    if 'forgot_password_flow' not in st.session_state:
        st.session_state['forgot_password_flow'] = False

    if not st.session_state['logged_in']:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="margin-bottom: 10px;">👗 Fashion Recommender</h1>
            <p style="font-size: 1.1rem; color: #666;">Discover your perfect style with AI-powered recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state['forgot_password_flow']:
            handle_forgot_password()
        else:
            if st.session_state['page'] == 'login':
                login()
            else:
                signup()
    else:
        user_details_sidebar()
        recommendation_system()

if __name__ == "__main__":
    main()
