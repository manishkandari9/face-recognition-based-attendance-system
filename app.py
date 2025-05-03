import cv2
import os
import shutil
from flask import Flask, request, render_template, send_file, Response, session, redirect, url_for, flash, jsonify
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import base64
import io
from PIL import Image

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'super_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create necessary directories
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# User Model for SQLite
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.String(20), unique=True, nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Utility Functions
def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)
    except Exception as e:
        logger.error(f"Error in identify_face: {str(e)}")
        return None

def train_model():
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
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance(selected_date=datetoday):
    try:
        df = pd.read_csv(f'Attendance/Attendance-{selected_date}.csv')
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
    except FileNotFoundError:
        names, rolls, times, l = [], [], [], 0
    return names, rolls, times, l

def add_attendance(name, selected_date=datetoday):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    if f'Attendance-{selected_date}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/Attendance-{selected_date}.csv', 'w') as f:
            f.write('Name,Roll,Time')
    df = pd.read_csv(f'Attendance/Attendance-{selected_date}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{selected_date}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)
    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, l

def deletefolder(duser):
    if os.path.exists(duser):
        pics = os.listdir(duser)
        for i in pics:
            os.remove(duser+'/'+i)
        os.rmdir(duser)

def get_attendance_dates():
    files = os.listdir('Attendance')
    dates = [f.split('-')[1].split('.')[0] for f in files if f.startswith('Attendance-')]
    return sorted(dates, reverse=True)

# Convert base64 image to OpenCV format
def base64_to_image(base64_string):
    try:
        img_data = base64.b64decode(base64_string.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error converting base64 to image: {str(e)}")
        return None

# Authentication Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
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
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user:
            if user.role == 'student' and check_password_hash(user.password, user.user_id):
                if password == user.user_id:
                    session['username'] = username
                    session['role'] = user.role
                    flash('Login successful!', 'success')
                    return redirect(url_for('student_dashboard'))
            elif user.role in ['admin', 'teacher'] and check_password_hash(user.password, password):
                session['username'] = username
                session['role'] = user.role
                flash('Login successful!', 'success')
                if user.role == 'admin':
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('teacher_dashboard'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

# Dashboard Routes
@app.route('/student')
def student_dashboard():
    if 'username' not in session or session['role'] != 'student':
        flash('Please log in as a student.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    selected_date = request.args.get('selected_date', datetoday)
    names, rolls, times, l = extract_attendance(selected_date)
    user_attendance = [(n, t) for n, r, t in zip(names, rolls, times) if str(r) == user.user_id]
    return render_template('student_dashboard.html', name=user.name, id=user.user_id,
                         attendance=user_attendance, totalreg=totalreg(), datetoday2=datetoday2,
                         dates=get_attendance_dates(), selected_date=selected_date)

@app.route('/teacher')
def teacher_dashboard():
    if 'username' not in session or session['role'] != 'teacher':
        flash('Please log in as a teacher.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    selected_date = request.args.get('selected_date', datetoday)
    names, rolls, times, l = extract_attendance(selected_date)
    return render_template('teacher_dashboard.html', name=user.name, id=user.user_id,
                         names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                         datetoday2=datetoday2, dates=get_attendance_dates(), selected_date=selected_date)

@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    selected_date = request.args.get('selected_date', datetoday)
    names, rolls, times, l = extract_attendance(selected_date)
    userlist, unames, urolls, ul = getallusers()
    return render_template('admin_dashboard.html', name=user.name, id=user.user_id,
                         names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                         datetoday2=datetoday2, dates=get_attendance_dates(), selected_date=selected_date,
                         userlist=userlist, unames=unames, urolls=urolls, ul=ul)

# Existing Routes
@app.route('/')
def home():
    if 'username' in session:
        if session['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif session['role'] == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        else:
            return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))

@app.route('/listusers')
def listusers():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls,
                         l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
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
    return redirect(url_for('listusers'))

@app.route('/download', methods=['GET'])
def download():
    if 'username' not in session or session['role'] not in ['teacher', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    selected_date = request.args.get('selected_date', datetoday)
    file_path = f'Attendance/Attendance-{selected_date}.csv'
    try:
        return send_file(file_path, as_attachment=True,
                        download_name=f'Attendance-{selected_date}.csv')
    except FileNotFoundError:
        flash('No attendance data for selected date.', 'danger')
        return redirect(url_for('teacher_dashboard' if session['role'] == 'teacher' else 'admin_dashboard'))

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        frame = base64_to_image(data['frame'])
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        selected_date = data.get('selected_date', datetoday)
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if identified_person:
                person_id = identified_person[0]
                username, userid = person_id.split('_')
                add_attendance(person_id, selected_date)
                return jsonify({
                    'person': person_id,
                    'name': username,
                    'roll': userid,
                    'box': {'x': x, 'y': y, 'w': w, 'h': h}
                })
        return jsonify({'person': None})
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/start', methods=['GET'])
def start():
    if 'username' not in session or session['role'] not in ['teacher', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    selected_date = request.args.get('selected_date', datetoday)
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        flash('No trained model found. Please add a new face to continue.', 'danger')
        return redirect(url_for('admin_dashboard'))
    # Local system par purana flow (cv2.imshow ke saath)
    if os.getenv('RENDER') is None:  # Local environment
        ret = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            flash('Unable to access webcam!', 'danger')
            return redirect(url_for('admin_dashboard'))
        while ret:
            ret, frame = cap.read()
            if not ret or frame is None:
                flash('Error capturing video!', 'danger')
                break
            if len(extract_faces(frame)) > 0:
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))
                if identified_person:
                    username, userid = identified_person[0].split('_')
                    add_attendance(identified_person[0], selected_date)
                    cv2.putText(frame, f'{username} ({userid})', (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:  # ESC key to exit
                break
        cap.release()
        cv2.destroyAllWindows()
        flash('Attendance recorded successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    # Server (Render) par WebRTC-based flow
    else:
        return render_template('start_attendance.html', selected_date=selected_date)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if 'username' not in session or session['role'] != 'admin':
        flash('Please log in as an admin.', 'danger')
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash('User not found. Please log in again.', 'danger')
        session.pop('username', None)
        session.pop('role', None)
        return redirect(url_for('login'))
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        is_modal = request.form.get('is_modal', 'false') == 'true'
        
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
        
        if os.getenv('RENDER') is None:  # Local environment
            i, j = 0, 0
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                flash('Unable to access webcam!', 'danger')
                if not existing_user:
                    db.session.delete(new_user)
                    db.session.commit()
                return redirect(url_for('admin_dashboard'))
            while True:
                ret, frame = cap.read()
                if not ret:
                    flash('Error capturing video!', 'danger')
                    if not existing_user:
                        db.session.delete(new_user)
                        db.session.commit()
                    cap.release()
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
                    return redirect(url_for('admin_dashboard'))
            cap.release()
            cv2.destroyAllWindows()
            try:
                train_model()
                flash('Student added and face data captured successfully!', 'success')
            except Exception as e:
                flash(f'Error training model: {str(e)}', 'danger')
                if not existing_user:
                    db.session.delete(new_user)
                    db.session.commit()
                deletefolder(userimagefolder)
            return redirect(url_for('admin_dashboard'))
        else:  # Server environment
            if is_modal:
                return jsonify({
                    'success': True,
                    'newusername': newusername,
                    'newuserid': newuserid,
                    'nimgs': nimgs
                })
            return render_template('add_face.html', newusername=newusername, newuserid=newuserid, nimgs=nimgs)
        
    return redirect(url_for('admin_dashboard'))

@app.route('/batch_capture', methods=['POST'])
def batch_capture():
    try:
        data = request.json
        frames = data['frames']
        newusername = data['newusername']
        newuserid = data['newuserid']
        
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        
        saved_count = 0
        for i, frame_data in enumerate(frames):
            frame = base64_to_image(frame_data)
            if frame is None:
                continue
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                name = f'{newusername}_{saved_count}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', face_img)
                saved_count += 1
                if saved_count >= nimgs:
                    break
        
        if saved_count >= nimgs:
            train_model()
            flash('Student added and face data captured successfully!', 'success')
            return jsonify({'success': True})
        else:
            shutil.rmtree(userimagefolder)
            return jsonify({'success': False, 'error': f'Captured only {saved_count} valid faces, need {nimgs}'}), 400
    except Exception as e:
        logger.error(f"Error in batch capture: {str(e)}")
        if os.path.exists(userimagefolder):
            shutil.rmtree(userimagefolder)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/finish_capture', methods=['POST'])
def finish_capture():
    try:
        data = request.json
        newusername = data['newusername']
        newuserid = data['newuserid']
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        
        if os.path.exists(userimagefolder):
            train_model()
            flash('Student added and face data captured successfully!', 'success')
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'No images captured'}), 400
    except Exception as e:
        logger.error(f"Error finishing capture: {str(e)}")
        if os.path.exists(userimagefolder):
            shutil.rmtree(userimagefolder)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
