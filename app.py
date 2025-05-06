import cv2
import os
import shutil
import logging
from flask import Flask, request, render_template, send_file, Response, session, redirect, url_for, flash
from datetime import date, datetime, time, timedelta
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'super_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

nimgs = 20  # Number of images for training

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create necessary directories
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('logs'):
    os.makedirs('logs')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Timestamps')

# User Model for SQLite
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.String(20), unique=True, nullable=False)

# Cutoff Time Model
class CutoffTime(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cutoff_hour = db.Column(db.Integer, nullable=False)
    cutoff_minute = db.Column(db.Integer, nullable=False)

# Morning Time Model
class MorningTime(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    morning_hour = db.Column(db.Integer, nullable=False)
    morning_minute = db.Column(db.Integer, nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Utility Functions
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
        return face_points
    except Exception as e:
        logging.error(f"Error in extract_faces: {str(e)}")
        return []

def identify_face(facearray, tolerance=0.5):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        prediction = model.predict(facearray)[0]
        confidence = model.predict_proba(facearray).max()
        logging.info(f"Predicted: {prediction}, Confidence: {confidence}")
        if confidence < tolerance:
            return "Unknown"
        return prediction
    except Exception as e:
        logging.error(f"Error in identify_face: {str(e)}")
        return "Unknown"

def train_model():
    try:
        faces = []
        labels = []
        userlist = os.listdir('static/faces')
        for user in userlist:
            for imgname in os.listdir(f'static/faces/{user}'):
                img = cv2.imread(f'static/faces/{user}/{imgname}')
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        logging.info("Model trained and saved successfully")
    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}")
        raise

def get_cutoff_time():
    try:
        cutoff = CutoffTime.query.first()
        if cutoff:
            return time(cutoff.cutoff_hour, cutoff.cutoff_minute)
        return time(15, 30)  # Default to 3:30 PM
    except Exception as e:
        logging.error(f"Error in get_cutoff_time: {str(e)}")
        return time(15, 30)

def get_morning_time():
    try:
        morning = MorningTime.query.first()
        if morning:
            return time(morning.morning_hour, morning.morning_minute)
        return time(9, 0)  # Default to 9:00 AM
    except Exception as e:
        logging.error(f"Error in get_morning_time: {str(e)}")
        return time(9, 0)

def format_time_12hr(hour, minute):
    """Convert 24-hour time to 12-hour format"""
    try:
        dt = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M")
        return dt.strftime("%I:%M %p")
    except Exception as e:
        logging.error(f"Error in format_time_12hr: {str(e)}")
        return "Invalid Time"

def parse_time_12hr(time_str):
    """Parse 12-hour time (e.g., '03:30 PM') to 24-hour hour, minute"""
    try:
        dt = datetime.strptime(time_str, "%I:%M %p")
        return dt.hour, dt.minute
    except ValueError as e:
        logging.error(f"Error in parse_time_12hr: {str(e)}")
        raise ValueError("Invalid time format. Use HH:MM AM/PM (e.g., 03:30 PM)")

def extract_attendance(selected_date=datetoday):
    try:
        df = pd.read_csv(f'Attendance/Attendance-{selected_date}.csv')
        names = df['Name'].tolist()
        rolls = df['Roll'].astype(str).tolist()
        times = df['Timestamps'].str.split(',', expand=True).stack().reset_index(drop=True).tolist()
        l = len(df)
        
        # Get cutoff and morning time
        cutoff_time = get_cutoff_time()
        morning_time = get_morning_time()
        current_time = datetime.now().time()
        
        # Initialize attendance status
        statuses = []
        processed_rolls = set()
        
        # Process users in attendance
        for roll in rolls:
            if roll in processed_rolls:
                continue
            processed_rolls.add(roll)
            user_idx = df[df['Roll'] == int(roll)].index
            if not user_idx.empty:
                timestamps = df.loc[user_idx[0], 'Timestamps'].split(',')
                timestamp_dts = [datetime.strptime(t, "%I:%M:%S %p") for t in timestamps]
                morning_attendance = any(morning_time <= dt.time() <= time(15, 0) for dt in timestamp_dts)  # Up to 3:00 PM
                cutoff_attendance = any(cutoff_time <= dt.time() <= time(17, 0) for dt in timestamp_dts)  # Up to 5:00 PM
                
                # Status logic
                if morning_attendance or cutoff_attendance:
                    if current_time > time(17, 0):  # After 5:00 PM, check for final status
                        if morning_attendance and not cutoff_attendance:
                            statuses.append("Left Early")
                        else:
                            statuses.append("Present")
                    else:
                        statuses.append("Present")  # During the day, mark as Present
                else:
                    statuses.append("Absent")
            else:
                statuses.append("Absent")
        
        return names, rolls, times, statuses, l
    except FileNotFoundError:
        logging.warning(f"No attendance file found for date: {selected_date}")
        return [], [], [], [], 0
    except Exception as e:
        logging.error(f"Error in extract_attendance: {str(e)}")
        return [], [], [], [], 0

def add_attendance(name, selected_date=datetoday):
    if name == "Unknown":
        logging.info("Skipping attendance for unknown face")
        return
    try:
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%I:%M:%S %p")  # 12-hour format
        current_dt = datetime.now()
        
        # Allow attendance only during morning time to 3:00 PM or cutoff time to 5:00 PM
        morning_time = get_morning_time()
        cutoff_time = get_cutoff_time()
        morning_end = time(15, 0)  # 3:00 PM
        cutoff_end = time(17, 0)   # 5:00 PM
        
        if not ((morning_time <= current_dt.time() <= morning_end) or (cutoff_time <= current_dt.time() <= cutoff_end)):
            logging.info(f"Attendance attempt outside allowed time: {current_dt.time()}")
            return
        
        if f'Attendance-{selected_date}.csv' not in os.listdir('Attendance'):
            with open(f'Attendance/Attendance-{selected_date}.csv', 'w') as f:
                f.write('Name,Roll,Timestamps')
        
        df = pd.read_csv(f'Attendance/Attendance-{selected_date}.csv')
        # Check if user already exists in CSV
        user_row = df[df['Roll'] == int(userid)]
        if not user_row.empty:
            # Append new timestamp to existing Timestamps
            current_timestamps = user_row['Timestamps'].iloc[0]
            if current_time not in current_timestamps.split(','):
                new_timestamps = f"{current_timestamps},{current_time}" if current_timestamps else current_time
                df.loc[df['Roll'] == int(userid), 'Timestamps'] = new_timestamps
        else:
            # Add new user entry
            new_row = pd.DataFrame({'Name': [username], 'Roll': [int(userid)], 'Timestamps': [current_time]})
            df = pd.concat([df, new_row], ignore_index=True)
        
        # Save updated CSV
        df.to_csv(f'Attendance/Attendance-{selected_date}.csv', index=False)
        logging.info(f"Attendance added for {username} ({userid}) at {current_time}")
    except Exception as e:
        logging.error(f"Error in add_attendance: {str(e)}")

def getallusers():
    try:
        userlist = os.listdir('static/faces')
        names = []
        rolls = []
        l = len(userlist)
        for i in userlist:
            name, roll = i.split('_')
            names.append(name)
            rolls.append(roll)
        return userlist, names, rolls, l
    except Exception as e:
        logging.error(f"Error in getallusers: {str(e)}")
        return [], [], [], 0

def deletefolder(duser):
    try:
        if os.path.exists(duser):
            pics = os.listdir(duser)
            for i in pics:
                os.remove(duser+'/'+i)
            os.rmdir(duser)
            logging.info(f"Deleted folder: {duser}")
    except Exception as e:
        logging.error(f"Error in deletefolder: {str(e)}")

def get_attendance_dates():
    try:
        files = os.listdir('Attendance')
        dates = [f.split('-')[1].split('.')[0] for f in files if f.startswith('Attendance-')]
        return sorted(dates, reverse=True)
    except Exception as e:
        logging.error(f"Error in get_attendance_dates: {str(e)}")
        return []

def clean_faces_folder():
    try:
        db_users = {f"{user.name}_{user.user_id}" for user in User.query.all()}
        face_folders = set(os.listdir('static/faces'))
        for folder in face_folders - db_users:
            deletefolder(f'static/faces/{folder}')
        logging.info("Cleaned faces folder")
    except Exception as e:
        logging.error(f"Error in clean_faces_folder: {str(e)}")

# Authentication Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form['username'].lower().replace(' ', '_')
            password = request.form['password']
            name = request.form['name']
            user_id = request.form['user_id']
            role = request.form['role']
            
            if role not in ['admin', 'teacher']:
                flash('Invalid role selected. Only admins and teachers can sign up here.', 'danger')
                return redirect(url_for('signup'))
            
            if User.query.filter_by(username=username).first():
                flash('Username already exists!', 'danger')
                return redirect(url_for('signup'))
            
            if User.query.filter_by(user_id=user_id).first():
                flash('User ID already exists!', 'danger')
                return redirect(url_for('signup'))
            
            new_user = User(
                username=username,
                password=generate_password_hash(password),
                role=role,
                name=name,
                user_id=user_id
            )
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            logging.info(f"New user signed up: {username} ({role})")
            return redirect(url_for('login'))
        except Exception as e:
            logging.error(f"Error in signup: {str(e)}")
            flash('Error during signup. Please try again.', 'danger')
            return redirect(url_for('signup'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            user = User.query.filter_by(username=username).first()
            if user:
                if user.role == 'student' and check_password_hash(user.password, user.user_id):
                    if password == user.user_id:
                        session['username'] = username
                        session['role'] = user.role
                        flash('Login successful!', 'success')
                        logging.info(f"Student login: {username}")
                        return redirect(url_for('student_dashboard'))
                elif user.role in ['admin', 'teacher'] and check_password_hash(user.password, password):
                    session['username'] = username
                    session['role'] = user.role
                    flash('Login successful!', 'success')
                    logging.info(f"{user.role.capitalize()} login: {username}")
                    if user.role == 'admin':
                        return redirect(url_for('admin_dashboard'))
                    else:
                        return redirect(url_for('teacher_dashboard'))
            flash('Invalid credentials', 'danger')
            logging.warning(f"Failed login attempt for username: {username}")
        except Exception as e:
            logging.error(f"Error in login: {str(e)}")
            flash('Error during login. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    try:
        username = session.get('username', 'Unknown')
        session.pop('username', None)
        session.pop('role', None)
        flash('Logged out successfully!', 'success')
        logging.info(f"User logged out: {username}")
        return redirect(url_for('login'))
    except Exception as e:
        logging.error(f"Error in logout: {str(e)}")
        flash('Error during logout. Please try again.', 'danger')
        return redirect(url_for('login'))

# Dashboard Routes
@app.route('/student')
def student_dashboard():
    if 'username' not in session or session['role'] != 'student':
        flash('Please log in as a student.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        selected_date = request.args.get('selected_date', datetoday)
        names, rolls, times, statuses, l = extract_attendance(selected_date)
        user_attendance = [(n, t, s) for n, r, t, s in zip(names, rolls, times, statuses) if str(r) == user.user_id]
        return render_template('student_dashboard.html', name=user.name, id=user.user_id,
                             attendance=user_attendance, totalreg=totalreg(), datetoday2=datetoday2,
                             dates=get_attendance_dates(), selected_date=selected_date)
    except Exception as e:
        logging.error(f"Error in student_dashboard: {str(e)}")
        flash('Error loading dashboard. Please try again.', 'danger')
        return redirect(url_for('login'))

@app.route('/teacher')
def teacher_dashboard():
    if 'username' not in session or session['role'] != 'teacher':
        flash('Please log in as a teacher.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        selected_date = request.args.get('selected_date', datetoday)
        names, rolls, times, statuses, l = extract_attendance(selected_date)
        return render_template('teacher_dashboard.html', name=user.name, id=user.user_id,
                             names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(),
                             datetoday2=datetoday2, dates=get_attendance_dates(), selected_date=selected_date)
    except Exception as e:
        logging.error(f"Error in teacher_dashboard: {str(e)}")
        flash('Error loading dashboard. Please try again.', 'danger')
        return redirect(url_for('login'))

@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        selected_date = request.args.get('selected_date', datetoday)
        names, rolls, times, statuses, l = extract_attendance(selected_date)
        userlist, unames, urolls, ul = getallusers()
        cutoff = CutoffTime.query.first()
        morning = MorningTime.query.first()
        cutoff_str = format_time_12hr(cutoff.cutoff_hour, cutoff.cutoff_minute) if cutoff else "03:30 PM"
        morning_str = format_time_12hr(morning.morning_hour, morning.morning_minute) if morning else "09:00 AM"
        return render_template('admin_dashboard.html', name=user.name, id=user.user_id,
                             names=names, rolls=rolls, times=times, statuses=statuses, l=l, totalreg=totalreg(),
                             datetoday2=datetoday2, dates=get_attendance_dates(), selected_date=selected_date,
                             userlist=userlist, unames=unames, urolls=urolls, ul=ul, cutoff_time=cutoff_str,
                             morning_time=morning_str)
    except Exception as e:
        logging.error(f"Error in admin_dashboard: {str(e)}")
        flash('Error loading dashboard. Please try again.', 'danger')
        return redirect(url_for('login'))

@app.route('/set_cutoff', methods=['GET', 'POST'])
def set_cutoff():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        
        if request.method == 'POST':
            cutoff_time = request.form['cutoff_time']
            try:
                hour, minute = parse_time_12hr(cutoff_time)
                cutoff = CutoffTime.query.first()
                if cutoff:
                    cutoff.cutoff_hour = hour
                    cutoff.cutoff_minute = minute
                else:
                    cutoff = CutoffTime(cutoff_hour=hour, cutoff_minute=minute)
                    db.session.add(cutoff)
                db.session.commit()
                flash('Cutoff time updated successfully!', 'success')
                logging.info(f"Cutoff time set to: {cutoff_time}")
            except ValueError:
                flash('Invalid time format. Use HH:MM AM/PM (e.g., 03:30 PM).', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        cutoff = CutoffTime.query.first()
        cutoff_str = format_time_12hr(cutoff.cutoff_hour, cutoff.cutoff_minute) if cutoff else "03:30 PM"
        return render_template('set_cutoff.html', cutoff_time=cutoff_str)
    except Exception as e:
        logging.error(f"Error in set_cutoff: {str(e)}")
        flash('Error setting cutoff time. Please try again.', 'danger')
        return redirect(url_for('admin_dashboard'))

@app.route('/set_morning_time', methods=['GET', 'POST'])
def set_morning_time():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        
        if request.method == 'POST':
            morning_time = request.form['morning_time']
            try:
                hour, minute = parse_time_12hr(morning_time)
                morning = MorningTime.query.first()
                if morning:
                    morning.morning_hour = hour
                    morning.morning_minute = minute
                else:
                    morning = MorningTime(morning_hour=hour, morning_minute=minute)
                    db.session.add(morning)
                db.session.commit()
                flash('Morning time updated successfully!', 'success')
                logging.info(f"Morning time set to: {morning_time}")
            except ValueError:
                flash('Invalid time format. Use HH:MM AM/PM (e.g., 09:00 AM).', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        morning = MorningTime.query.first()
        morning_str = format_time_12hr(morning.morning_hour, morning.morning_minute) if morning else "09:00 AM"
        return render_template('set_morning_time.html', morning_time=morning_str)
    except Exception as e:
        logging.error(f"Error in set_morning_time: {str(e)}")
        flash('Error setting morning time. Please try again.', 'danger')
        return redirect(url_for('admin_dashboard'))

# Existing Routes
@app.route('/')
def home():
    try:
        if 'username' in session:
            if session['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif session['role'] == 'teacher':
                return redirect(url_for('teacher_dashboard'))
            else:
                return redirect(url_for('student_dashboard'))
        return redirect(url_for('login'))
    except Exception as e:
        logging.error(f"Error in home: {str(e)}")
        flash('Error redirecting to home. Please try again.', 'danger')
        return redirect(url_for('login'))

@app.route('/listusers')
def listusers():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        userlist, names, rolls, l = getallusers()
        return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls,
                             l=l, totalreg=totalreg(), datetoday2=datetoday2)
    except Exception as e:
        logging.error(f"Error in listusers: {str(e)}")
        flash('Error loading user list. Please try again.', 'danger')
        return redirect(url_for('admin_dashboard'))

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        
        duser = request.args.get('user')
        if not duser:
            flash('No user specified for deletion.', 'danger')
            return redirect(url_for('listusers'))
        
        try:
            username, user_id = duser.split('_')
        except ValueError:
            flash('Invalid user format.', 'danger')
            return redirect(url_for('listusers'))
        
        user_to_delete = User.query.filter_by(username=username, user_id=user_id).first()
        if user_to_delete:
            db.session.delete(user_to_delete)
            db.session.commit()
        else:
            flash('User not found in database.', 'warning')
        
        user_folder = f'static/faces/{duser}'
        if os.path.exists(user_folder):
            try:
                deletefolder(user_folder)
            except Exception as e:
                flash(f'Error deleting face images: {str(e)}', 'danger')
                return redirect(url_for('listusers'))
        
        if os.listdir('static/faces'):
            try:
                train_model()
            except Exception as e:
                flash(f'Error retraining model: {str(e)}', 'warning')
        else:
            if os.path.exists('static/face_recognition_model.pkl'):
                os.remove('static/face_recognition_model.pkl')
        
        flash('User deleted successfully!', 'success')
        logging.info(f"User deleted: {duser}")
        return redirect(url_for('listusers'))
    except Exception as e:
        logging.error(f"Error in deleteuser: {str(e)}")
        flash('Error deleting user. Please try again.', 'danger')
        return redirect(url_for('listusers'))

@app.route('/download', methods=['GET'])
def download():
    if 'username' not in session or session['role'] not in ['teacher', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        selected_date = request.args.get('selected_date', datetoday)
        file_path = f'Attendance/Attendance-{selected_date}.csv'
        return send_file(file_path, as_attachment=True,
                        download_name=f'Attendance-{selected_date}.csv')
    except FileNotFoundError:
        flash('No attendance data for selected date.', 'danger')
        logging.warning(f"Attendance file not found: {file_path}")
        return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))
    except Exception as e:
        logging.error(f"Error in download: {str(e)}")
        flash('Error downloading attendance. Please try again.', 'danger')
        return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))

@app.route('/start', methods=['GET'])
def start():
    if 'username' not in session or session['role'] not in ['teacher', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        selected_date = request.args.get('selected_date', datetoday)
        names, rolls, times, statuses, l = extract_attendance(selected_date)
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            flash('No trained model found. Please add a new face to continue.', 'danger')
            return redirect(url_for('admin_dashboard'))
        
        # Check if current time is within allowed attendance periods
        current_dt = datetime.now()
        morning_time = get_morning_time()
        cutoff_time = get_cutoff_time()
        morning_end = time(15, 0)  # 3:00 PM
        cutoff_end = time(17, 0)   # 5:00 PM
        
        logging.debug(f"Current Time: {current_dt.time()}, Morning Time: {morning_time}, Cutoff Time: {cutoff_time}")
        
        if not ((morning_time <= current_dt.time() <= morning_end) or (cutoff_time <= current_dt.time() <= cutoff_end)):
            flash('Attendance can only be taken from morning time to 3:00 PM or from cutoff time to 5:00 PM.', 'danger')
            logging.info(f"Attendance attempt outside allowed time: {current_dt.time()}")
            return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))
        
        ret = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash('Unable to access webcam!', 'danger')
            logging.error("Webcam not accessible")
            return redirect(url_for('admin_dashboard'))
        last_attendance_time = None
        while ret:
            ret, frame = cap.read()
            if not ret:
                flash('Error capturing video!', 'danger')
                logging.error("Error capturing video from webcam")
                cap.release()
                cv2.destroyAllWindows()
                return redirect(url_for('admin_dashboard'))
            if len(extract_faces(frame)) > 0:
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1), tolerance=0.5)
                if identified_person != "Unknown":
                    current_time = datetime.now().strftime("%I:%M:%S %p")
                    # Only add attendance if at least 1 second has passed
                    if last_attendance_time is None or (datetime.now() - datetime.strptime(last_attendance_time, "%I:%M:%S %p")).total_seconds() >= 1:
                        add_attendance(identified_person, selected_date)
                        last_attendance_time = current_time
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(f'logs/detected_{identified_person}_{datetime.now().strftime("%H%M%S")}.jpg', frame)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        flash('Attendance recorded successfully!', 'success')
        logging.info("Attendance recording completed")
        return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))
    except Exception as e:
        logging.error(f"Error in start: {str(e)}")
        flash('Error recording attendance. Please try again.', 'danger')
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))

@app.route('/add', methods=['GET', 'POST'])
def add():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            flash('User not found. Please log in again.', 'danger')
            session.pop('username', None)
            session.pop('role', None)
            return redirect(url_for('login'))
        if request.method == 'POST':
            newusername = request.form['newusername']
            newuserid = request.form['newuserid']
            
            normalized_username = newuserid
            
            existing_user = User.query.filter_by(user_id=newuserid).first()
            
            if existing_user:
                userimagefolder = f'static/faces/{existing_user.name}_{newuserid}'
                if os.path.exists(userimagefolder):
                    shutil.rmtree(userimagefolder)
            else:
                if User.query.filter_by(username=normalized_username).first():
                    flash('User ID already exists as a username!', 'danger')
                    return redirect(url_for('admin_dashboard'))
                new_user = User(
                    username=normalized_username,
                    password=generate_password_hash(newuserid),
                    role='student',
                    name=newusername,
                    user_id=newuserid
                )
                db.session.add(new_user)
                db.session.commit()
                userimagefolder = f'static/faces/{newusername}_{newuserid}'
            
            if not os.path.isdir(userimagefolder):
                os.makedirs(userimagefolder)
            i, j = 0, 0
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                flash('Unable to access webcam!', 'danger')
                if not existing_user:
                    db.session.delete(new_user)
                    db.session.commit()
                logging.error("Webcam not accessible for adding user")
                return redirect(url_for('admin_dashboard'))
            while True:
                ret, frame = cap.read()
                if not ret:
                    flash('Error capturing video!', 'danger')
                    if not existing_user:
                        db.session.delete(new_user)
                        db.session.commit()
                    cap.release()
                    logging.error("Error capturing video for adding user")
                    return redirect(url_for('admin_dashboard'))
                faces = extract_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    if j % 5 == 0:
                        name = f'{newusername}_{i}.jpg'
                        cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                        i += 1
                    j += 1
                if j == nimgs*5:
                    break
                cv2.imshow('Adding new User', frame)
                if cv2.waitKey(1) == 27:
                    flash('Face capture cancelled.', 'warning')
                    if not existing_user:
                        db.session.delete(new_user)
                        db.session.commit()
                    deletefolder(userimagefolder)
                    cap.release()
                    cv2.destroyAllWindows()
                    logging.info("Face capture cancelled by user")
                    return redirect(url_for('admin_dashboard'))
            cap.release()
            cv2.destroyAllWindows()
            
            try:
                train_model()
                clean_faces_folder()
                flash('Student added and face data captured successfully!', 'success')
                logging.info(f"New student added: {newusername} ({newuserid})")
            except Exception as e:
                flash(f'Error training model: {str(e)}', 'danger')
                if not existing_user:
                    db.session.delete(new_user)
                    db.session.commit()
                deletefolder(userimagefolder)
                logging.error(f"Error training model for new user: {str(e)}")
                return redirect(url_for('admin_dashboard'))
        
        return redirect(url_for('admin_dashboard'))
    except Exception as e:
        logging.error(f"Error in add: {str(e)}")
        flash('Error adding new user. Please try again.', 'danger')
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)