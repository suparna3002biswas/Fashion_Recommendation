from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Upload configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Image model
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    filename = db.Column(db.String(300), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize DB
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!')
        return redirect(url_for('login'))

    return render_template('register.html')

# Route: Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()

        if user:
            login_user(user)
            session['username'] = user.username
            return redirect(url_for('upload_image'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

# Route: Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('username', None)
    flash('Logged out successfully!')
    return redirect(url_for('login'))

# Route: Upload image
@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file:
            username = session['username']
            
            # Create a folder for each user
            user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username)
            os.makedirs(user_folder, exist_ok=True)
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(user_folder, filename)
            
            file.save(filepath)

            # Store image details in the database
            new_image = Image(username=username, filename=filename)
            db.session.add(new_image)
            db.session.commit()

            flash('Image uploaded successfully!')
            return redirect(url_for('gallery'))

    return render_template('index.html')

# Route: Display only the user's images with timestamps
@app.route('/gallery')
@login_required
def gallery():
    username = session['username']

    # Fetch images for the current user
    images = Image.query.filter_by(username=username).all()

    return render_template('gallery.html', images=images)

# Route: Delete image
@app.route('/delete/<int:image_id>', methods=['POST'])
@login_required
def delete_image(image_id):
    image = Image.query.get(image_id)

    if image and image.username == session['username']:
        # Delete image from filesystem
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.username, image.filename)
        
        if os.path.exists(image_path):
            os.remove(image_path)

        # Remove image from the database
        db.session.delete(image)
        db.session.commit()

        flash('Image deleted successfully!')

    return redirect(url_for('gallery'))

if __name__ == '__main__':
    app.run(debug=True)
