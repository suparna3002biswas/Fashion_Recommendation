import streamlit as st
import os
import numpy as np
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
import json

# Custom CSS for styling
def set_custom_css():
    st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #ffd1dc 0%, #ffb6c1 50%, #ff8c9e 100%);
    }
    
    /* Sidebar styling - made more compact */
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px;
        padding: 15px !important;
        margin: 10px 5px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: auto !important;
        max-width: 300px !important;
    }
    
    /* Make sidebar contents fit better */
    .stSidebar .stContainer {
        padding: 8px !important;
    }
    
    /* Card-like containers - made slightly more prominent */
    .stContainer, .stExpander, .stFileUploader {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        color: #d6336c;
    }
    
    /* Button styling - applied to both st.button and st.form_submit_button */
    .stButton>button, .stFormSubmitButton>button {
        background-color: #ff6b81 !important;
        color: white !important;
        border: none;
        transition: all 0.3s;
        border-radius: 8px !important;
        padding: 8px 16px !important;
    }
    
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #ff4757 !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Input field labels - make them black */
    .stTextInput label, 
    .stTextInput p,
    .stPasswordInput label,
    .stPasswordInput p,
    .stNumberInput label,
    .stNumberInput p {
        color: black !important;
        font-size: 15px !important;
    }
    
    /* Header styling */
    h1 {
        color: black !important;
        margin-top: 0.5em !important;
        font-size: 40px !important;
        letter-spacing: 0.5px;
    }
    
    h2, h3, h4, h5, h6 {
        color: #d6336c !important;
        margin-top: 0.5em !important;
        font-size: 30px !important;
        letter-spacing: 0.5px;
    }
    
    
    /* Input fields */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        color: black !important;
    }
    
    
    /* Cursor color */
    input, textarea {
        caret-color: black !important;
    }
    
    /* Image containers */
    .stImage>img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    .stImage>img:hover {
        transform: scale(1.03);
    }
    
    /* Spinner color */
    .stSpinner>div>div {
        border-color: #ff6b81 transparent transparent transparent !important;
    }
    
    /* Make expanders more compact */
    .stExpander > div {
        padding: 8px 0 !important;
    }
    
    /* Adjust form elements in sidebar */
    .stSidebar .stForm {
        padding: 0 !important;
    }
    
    /* Ensure sidebar doesn't stretch vertically */
    [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    /* Consistent image container styling */
    .image-container {
        width: 100% !important;
        height: 250px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        overflow: hidden !important;
        margin-bottom: 10px !important;
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        cursor: pointer !important;
        transition: transform 0.2s !important;
    }
        
    .image-container:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
        
    /* Make images fill containers while maintaining aspect ratio */
    .image-container img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
        background-color: #f8f8f8 !important;
    }
        
    /* Dropdown pointer */
    div[data-baseweb="select"] div {
        cursor: pointer !important;
    }
    </style>
    """, unsafe_allow_html=True)
  
# Database setup - Updated to include user details
def create_users_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, 
                  password TEXT,
                  name TEXT,
                  age INTEGER,
                  alt_email TEXT)''')
                 
    c.execute('''CREATE TABLE IF NOT EXISTS password_reset
                 (email TEXT NOT NULL,
                  token TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY (email, token))''')
    
    conn.commit()
    conn.close()

def add_user(email, password, name=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (email, password, name) VALUES (?, ?, ?)", 
              (email, password, name))
    conn.commit()
    conn.close()

def update_user_details(email, name=None, age=None, alt_email=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    updates = []
    params = []
    
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if age is not None:
        updates.append("age = ?")
        params.append(age)
    if alt_email is not None:
        updates.append("alt_email = ?")
        params.append(alt_email)
    
    if updates:
        update_query = "UPDATE users SET " + ", ".join(updates) + " WHERE email = ?"
        params.append(email)
        c.execute(update_query, params)
    
    conn.commit()
    conn.close()

def get_user_details(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, age, alt_email FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    
    return {
        'name': result[0] if result else None,
        'age': result[1] if result else None,
        'alt_email': result[2] if result else None
    } if result else None

def check_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
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

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication functions - Updated to include name
def signup():
    with st.container():
        st.subheader("Create New Account")
        email = st.text_input("Email")
        name = st.text_input("Name (optional)")
        password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')
        
        if st.button("Sign Up", key="signup_btn"):
            if password == confirm_password:
                if user_exists(email):
                    st.error("Email already exists!")
                else:
                    hashed_password = hash_password(password)
                    add_user(email, hashed_password, name)
                    st.success("Account created successfully! Please login.")
                    st.session_state['page'] = 'login'
                    st.rerun()
            else:
                st.error("Passwords do not match!")

# Password reset functions
def generate_reset_token(email):
    # Delete existing tokens
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM password_reset WHERE email=?", (email,))
    
    # Generate new token
    token = secrets.token_hex(3).upper()  # 6-character code
    c.execute("INSERT INTO password_reset (email, token) VALUES (?, ?)", 
             (email, token))
    conn.commit()
    conn.close()
    return token

def verify_reset_token(email, token):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Check token validity (15 minutes expiration)
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

# Updated login function
def login():
    with st.container():
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        
        if st.button("Login", key="login_btn"):
            hashed_password = hash_password(password)
            if check_user(email, hashed_password):
                st.session_state['logged_in'] = True
                st.session_state['email'] = email
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

        # Password reset section
        if st.session_state.get('show_reset'):
            with st.form("password_reset_form"):
                st.subheader("Password Reset")
                code = st.text_input("Reset Code")
                new_pass = st.text_input("New Password", type='password')
                confirm_pass = st.text_input("Confirm Password", type='password')
                
                if st.form_submit_button("Reset Password"):
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

        if st.button("Forgot Password?"):
            st.session_state['forgot_password_flow'] = True
            st.rerun()

# New forgot password handler
def handle_forgot_password():
    with st.container():
        st.subheader("Forgot Password")
        email = st.text_input("Enter your registered email")
        
        if st.button("Send Reset Code"):
            if user_exists(email):
                token = generate_reset_token(email)
                st.session_state['reset_email'] = email
                st.session_state['show_reset'] = True
                st.success("Reset code sent to your email.")
                st.info(f"Debug: Your reset code is {token}")
            else:
                st.error("Email not found in our system")

        if st.session_state.get('show_reset'):
            with st.form("forgot_reset_form"):
                code = st.text_input("Enter Reset Code")
                new_pass = st.text_input("New Password", type='password')
                confirm_pass = st.text_input("Confirm Password", type='password')
                
                if st.form_submit_button("Reset Password"):
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

        if st.button("Back to Login"):
            st.session_state['forgot_password_flow'] = False
            st.rerun()

# User details form in sidebar
def user_details_sidebar():
    with st.sidebar:
        user_details = get_user_details(st.session_state['email'])
        
        if user_details is None:
            user_details = {'name': None, 'age': None, 'alt_email': None}
        
        display_name = user_details['name'] or st.session_state['email']
        st.title(f"👋 Welcome, {display_name}")
        
        with st.expander("📝 Update Your Profile"):
            with st.form("user_details_form"):
                name = st.text_input("Name", value=user_details['name'] or "")
                age = st.number_input("Age", min_value=1, max_value=120, 
                                    value=user_details['age'] or 18)
                alt_email = st.text_input("Alternative Email", 
                                        value=user_details['alt_email'] or "")
                
                if st.form_submit_button("💾 Update Details"):
                    update_user_details(
                        email=st.session_state['email'],
                        name=name if name else None,
                        age=age if age else None,
                        alt_email=alt_email if alt_email else None
                    )
                    st.success("Profile updated successfully!")
                    st.rerun()
        
        if st.button("🚪 Logout"):
            st.session_state['logged_in'] = False
            st.session_state['email'] = None
            st.rerun()

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

#Recommendation System
def recommendation_system():
    st.title("👗 Fashion Recommendation System")

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
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔍 Upload Image", "🎉 Find by Occasion"])
    
    # TAB 1: Upload Image
    with tab1:
        st.subheader("📤 Upload Your Fashion Item")
        uploaded_file = st.file_uploader("Choose an image", 
                                     type=['png', 'jpg', 'jpeg', 'webp', 'jfif'],
                                     key="uploader_tab1")
    
        if uploaded_file is not None:
            # Save uploaded file
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            uploaded_path = os.path.join(upload_dir, uploaded_file.name)
            with open(uploaded_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display uploaded image
            st.subheader("🖼️ Uploaded Image")
            st.image(uploaded_file, width=300)
            
            # Process image
            img = image.load_img(uploaded_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            
            # Predict category
            with st.spinner('🔍 Analyzing image...'):
                all_preds = [model.predict(img_preprocessed)[0] for model in models]
                avg_pred = np.mean(all_preds, axis=0)
                predicted_class_index = np.argmax(avg_pred)
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            
            st.subheader(f"🏷️ Predicted Category: {predicted_class}")
            
            # Find similar images
            with st.spinner('✨ Finding similar items...'):
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

                        if sim_score >= 0.8 and img_path != uploaded_path:
                            similar_images.append((img_path, sim_score))
                    except Exception as e:
                        continue

                similar_images.sort(key=lambda x: x[1], reverse=True)
                top_similar = similar_images[:10]

            # Display results
            st.subheader(f"💎 Recommended Similar Items ({len(top_similar)} found)")
            
            if len(top_similar) > 0:
                # Prepare image paths for slideshow
                image_paths = [path for path, _ in top_similar]
                image_paths_json = json.dumps(image_paths)
                
                cols = st.columns(4)
                for idx, (path, sim) in enumerate(top_similar):
                    with cols[idx % 4]:
                        # Clickable image container
                        st.markdown(
                            f"""
                            <div class="image-container" onclick="openSlideshow({image_paths_json}, {idx})">
                                <img src="data:image/png;base64,{image_to_base64(path)}">
                            </div>
                            <div class="image-caption">Similarity: {sim:.2f}</div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.warning("No similar items found above similarity threshold")

    # TAB 2: Find by Occasion
    with tab2:
        st.subheader("👔 Find Clothes by Occasion")
        
        occasion_options = ["Formal", "Casual", "Party", 
        "Wedding", "Common", "Winter", 
        "Daily", "Festive"
        ]

        selected_occasion = st.selectbox(
            "Select an occasion:", 
            occasion_options,
            key="occasion_select",
            help="Choose the occasion you're dressing for"
        )
        
        if st.button("Find Recommendations", key="occasion_btn"):
            with st.spinner(f'🔍 Finding {selected_occasion} outfits...'):
                # Find matching categories
                matching_categories = [
                    cat for cat in all_categories 
                    if selected_occasion.lower() in cat.lower()
                ]
                
                if not matching_categories:
                    from difflib import get_close_matches
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
                    st.success(f"Found matching category: {matched_category}")
                    
                    # Get category images
                    category_dir = os.path.join(dataset_path, matched_category)
                    category_images = [
                        os.path.join(category_dir, f)
                        for f in os.listdir(category_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', 'jfif'))
                        and not f.startswith('aug')
                    ]
                    
                    if category_images:
                        st.subheader(f"👗 {matched_category} Outfits for {selected_occasion}")
                        display_images = category_images[:min(10, len(category_images))]
                        
                        # Prepare image paths for slideshow
                        image_paths_json = json.dumps(display_images)
                        
                        # Display in 4-column grid
                        cols = st.columns(4)
                        for idx, img_path in enumerate(display_images):
                            with cols[idx % 4]:
                                # Clickable image container
                                st.markdown(
                                    f"""
                                    <div class="image-container" onclick="openSlideshow({image_paths_json}, {idx})">
                                        <img src="data:image/png;base64,{image_to_base64(img_path)}">
                                    </div>
                                    <div class="image-caption">{os.path.basename(img_path)}</div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.warning(f"No images found in the {matched_category} category")
                else:
                    st.error(f"Couldn't find a matching category for '{selected_occasion}'")
                    st.info("Available categories: " + ", ".join(all_categories[:10]) + "...")

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
        st.title("👗 Fashion Recommendation System")
        
        if st.session_state['forgot_password_flow']:
            handle_forgot_password()
        else:
            if st.session_state['page'] == 'login':
                login()
                if st.button("Don't have an account? Sign Up"):
                    st.session_state['page'] = 'signup'
                    st.rerun()
            else:
                signup()
                if st.button("Already have an account? Login"):
                    st.session_state['page'] = 'login'
                    st.rerun()
    else:
        user_details_sidebar()
        recommendation_system()

if __name__ == "__main__":
    main()