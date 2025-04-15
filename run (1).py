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
        color: #d6336c !important;
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
    </style>
    """, unsafe_allow_html=True)

# Database setup - Updated to include user details
def create_users_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create initial table structure
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, 
                  password TEXT)''')
    
    # Check and add missing columns
    c.execute("PRAGMA table_info(users)")
    existing_columns = [col[1] for col in c.fetchall()]
    
    # Define required columns
    required_columns = [
        ('name', 'TEXT'),
        ('age', 'INTEGER'),
        ('alt_email', 'TEXT')
    ]
    
    for col_name, col_type in required_columns:
        if col_name not in existing_columns:
            c.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
    
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
                st.error("Invalid email or password")

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

# Recommendation System (unchanged)
def recommendation_system():
    st.title("👗 Fashion Recommendation System")
    
    @st.cache_resource
    def load_models():
        model1 = load_model('/content/Models/personalized_fashion_densenet201.keras')
        model2 = load_model('/content/Models/personalized_fashion_mobilenet_v2.keras')
        model3 = load_model('/content/Models/personalized_fashion_Xception.keras')
        feature_extractor = Model(inputs=model1.input, outputs=model1.layers[-3].output)
        return [model1, model2, model3], feature_extractor

    @st.cache_resource
    def get_label_encoder(dataset_path):
        labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return label_encoder
    
    models, feature_extractor = load_models()
    dataset_path = "/content/Fashion_Recommendation/Fashion_Reccomendation1"
    label_encoder = get_label_encoder(dataset_path)
    
    with st.container():
        st.subheader("📤 Upload Your Fashion Item")
        uploaded_file = st.file_uploader("Choose an image", 
                                     type=['png', 'jpg', 'jpeg', 'webp', 'jfif'],
                                     label_visibility="collapsed")
    
    if uploaded_file is not None:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        uploaded_path = os.path.join(upload_dir, uploaded_file.name)
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.container():
            st.subheader("🖼️ Uploaded Image")
            st.image(uploaded_file, width=300)
            
            img = image.load_img(uploaded_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            
            with st.spinner('🔍 Analyzing image...'):
                all_preds = [model.predict(img_preprocessed)[0] for model in models]
                avg_pred = np.mean(all_preds, axis=0)
                predicted_class_index = np.argmax(avg_pred)
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            
            st.subheader(f"🏷️ Predicted Category: {predicted_class}")
            
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

            st.subheader(f"💎 Recommended Similar Items ({len(top_similar)} found)")
            
            if len(top_similar) > 0:
                cols = st.columns(4)
                for idx, (path, sim) in enumerate(top_similar):
                    with cols[idx % 4]:
                        st.image(path, use_container_width=True)
                        st.progress(float(sim))
                        st.caption(f"Similarity: {sim:.2f}")
            else:
                st.warning("No similar items found above similarity threshold")

# Main App
def main():
    set_custom_css()  # Apply custom CSS
    create_users_db()
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    
    if not st.session_state['logged_in']:
        st.title("👗 Fashion Recommendation System")
        
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
        # Show user details sidebar
        user_details_sidebar()
        
        # Show recommendation system
        recommendation_system()

if __name__ == "__main__":
    main()