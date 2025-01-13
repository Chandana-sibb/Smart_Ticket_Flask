from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify,Response
from pymongo import MongoClient
from datetime import datetime
import face_recognition
import face_recognition_models
import numpy as np
import cv2
import os
import qrcode
import io
import base64

app = Flask(__name__)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["user"]
collection = db["user_collection"]


# Folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/recognized_names.txt')
def serve_recognized_names():
    return send_file('recognized_names.txt', as_attachment=False)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/register')
def index2():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    mobile = request.form['mobile']
    aadhar = request.form['aadhar']
    photo = request.files['photo']

    # Check if Aadhar number already exists
    if collection.find_one({"aadhar": aadhar}):
        return "Aadhar number already registered!", 400

    # Save photo to the uploads folder
    photo_filename = f"{photo.filename}"
    photo_path = os.path.join(UPLOAD_FOLDER, photo_filename)
    photo.save(photo_path)

    # Save photo as Base64 for database storage
    with open(photo_path, "rb") as photo_file:
        photo_base64 = base64.b64encode(photo_file.read()).decode('utf-8')

    # Insert user data into MongoDB
    user_data = {
        "name": name,
        "mobile": mobile,
        "aadhar": aadhar,
        "photo": photo_base64,
        "qr_code": None,  # Placeholder for QR code
        "balance": 150,     # Initial balance
        "source": None,   # To be set when user scans their source station
        "destination": None, # To be set after user scans destination
        "fare":0 ,
        "count":0   
        
    }
    collection.insert_one(user_data)

    return "Registration successful!", 200

@app.route('/generate_qr')
def generate_qr_form():
    return render_template('generate_qr.html')

@app.route('/generate_qr', methods=['POST'])
def generate_qr():
    aadhar = request.form.get('aadhar')

    # Check if Aadhar number exists
    user = collection.find_one({"aadhar": aadhar})
    if not user:
        return "User not found. Please register first.", 400

    # Check if QR code already generated
    if user["qr_code"]:
        return redirect(url_for('view_qr', aadhar=aadhar))

    # Generate QR code
    qr = qrcode.QRCode()
    qr.add_data(aadhar)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')

    # Save QR code as Base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Update user document with QR code
    collection.update_one({"aadhar": aadhar}, {"$set": {"qr_code": qr_code_base64}})

    return redirect(url_for('view_qr', aadhar=aadhar))

@app.route('/view_qr')
def view_qr_():
    return render_template('view_qr.html')

@app.route('/view_qr/<aadhar>')
def view_qr(aadhar):
    # Fetch user by Aadhar number
    user = collection.find_one({"aadhar": aadhar})
    if not user or not user["qr_code"]:
        return "QR code not found. Please generate it first.", 400

    # Decode Base64 QR code to render as an image
    qr_code_data = user["qr_code"]
    return render_template('view_qr.html', qr_code_data=qr_code_data, aadhar=aadhar)

@app.route('/download_qr/<aadhar>')
def download_qr(aadhar):
    # Fetch user by Aadhar number
    user = collection.find_one({"aadhar": aadhar})
    if not user or not user["qr_code"]:
        return "QR code not found. Please generate it first.", 400

    # Convert Base64 QR code back to image
    qr_code_base64 = user["qr_code"]
    qr_code_data = base64.b64decode(qr_code_base64)
    buffer = io.BytesIO(qr_code_data)
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name=f"{aadhar}_qr_code.png")

# Load known face encodings and their names
image_folder = "uploads"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].replace("_", " ")
            known_face_names.append(name)
        else:
            print(f"No faces found in {filename}. Skipping...")

# Store recognized names
recognized_names = set()

# Route for home page
@app.route('/face_recognition')
def face_recognition1():
    return render_template("face_recognition.html")



face_recognition_model = "hog" 
@app.route('/start_recognition', methods=['GET'])
def start_recognition():
    def gen_frames():
        video_capture = cv2.VideoCapture(0)
        
        frame_counter = 0  # To skip frames for efficiency
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Skip every other frame to reduce processing time
            frame_counter += 1
            if frame_counter % 2 == 0:  # Process every 2nd frame
                continue

            # Resize and convert the frame
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces in the frame (using HOG for speed)
            face_locations = face_recognition.face_locations(rgb_small_frame, model=face_recognition_model)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    recognized_names.add(name)

                face_names.append(name)

            # Draw rectangles and labels
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Multiply by 4 to get the correct size for the original frame
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Encode frame and yield as a response
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        video_capture.release()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_names', methods=['GET'])
def get_recognized_names():
    # Check if there are any recognized names
    if recognized_names:
        # Check if any of the recognized names exist in the database
        matched_names = []
        for name in recognized_names:
            user_data = collection.find_one({"name": name})  # Searching by name
            if user_data:
                matched_names.append(name)

        if matched_names:
            return jsonify({
                "match_found": True,
                "message": f"Recognized names matched: {', '.join(matched_names)}"
            })
        else:
            return jsonify({
                "match_found": False,
                "message": "No recognized names found in the database."
            })
    else:
        return jsonify({
            "match_found": False,
            "message": "No recognized names available."
        })

@app.route('/save_recognized_names', methods=['POST'])
def save_recognized_names():
    # List to store only matched names
    matched_names_to_save = []

    # Loop through the recognized names and match them with the database
    for name in recognized_names:
        # Check if the recognized name exists in the database
        user_data = collection.find_one({"name": name})
        if user_data:
            matched_names_to_save.append(name)

    # If there are matched names, save them to the file
    if matched_names_to_save:
        with open("recognized_names.txt", "w") as file:
            for name in matched_names_to_save:
                file.write(f"{name}\n")
        return jsonify({"message": "Matched names saved successfully!"})
    else:
        return jsonify({"message": "No matched names found to save."})
    
@app.route('/scan_qr')
def scan_qr():
    return render_template("scan_qr.html")
    
@app.route('/check_recognized_names')
def check_recognized_names():
    # Path to your recognized_names.txt file
    recognized_names_file = 'recognized_names.txt'
    
    if os.path.exists(recognized_names_file):
        with open(recognized_names_file, 'r') as file:
            recognized_names = file.read().strip()  # Read and remove extra spaces or newline characters
            
            # If the file is empty, return a message indicating no recognized faces
            if recognized_names == "":
                return jsonify({"status": "No recognized faces found"})
            else:
                return jsonify({"status": "Faces recognized", "names": recognized_names})
    else:
        return jsonify({"status": "File not found", "error": "recognized_names.txt not found"})
    
# Route to handle the QR data and fetch the associated name
@app.route('/get_name_from_qr', methods=['GET'])
def get_name_from_qr():
    # Get the QR data sent from the frontend
    qr_data = request.args.get('qr_data')  # This will capture the query parameter 'qr_data'

    if qr_data:
        # Query MongoDB collection for a document with the 'qr_code_data' field matching qr_data
        user = collection.find_one({"aadhar": qr_data})  # Replace 'qr_code_data' with your actual field name
        
        if user:
            # If a user is found, return the name as JSON
            return jsonify({'name': user.get('name')})  # Replace 'name' with the actual field for the name
        else:
            # If no user found, return a response indicating that
            return jsonify({'name': None, 'message': 'No user found with this QR code data.'})
    else:
        # If qr_data is not provided in the request, return an error response
        return jsonify({'error': 'QR data is required'}), 400

def calculate_fare(source, destination):
    # Example distance-based fare calculation
    fare_chart = {
        ("Location1", "Location2"): 20,
        ("Location1", "Location3"): 30,
        ("Location1", "Location4"): 40,
        ("Location2", "Location3"): 15,
        ("Location2", "Location4"): 25,
        ("Location3", "Location4"): 10,
        # Add other combinations
    }
    return fare_chart.get((source, destination), 50)  # Default fare



@app.route('/save_trip', methods=['POST'])
def save_trip():
    data = request.json
    aadhar = data.get('aadhar')
    source = data.get('source')
    destination = data.get('destination')

    if not aadhar or not source or not destination:
        return jsonify({"error": "Missing required fields."}), 400

    if source == destination:
        return jsonify({"error": "Source and destination cannot be the same."}), 400

    # Fare calculation logic
    fare = calculate_fare(source, destination)

    # Find user and update balance and travel count
    user = collection.find_one({"aadhar": aadhar})
    if not user:
        return jsonify({"error": "User not found."}), 404

    if user["balance"] < fare:
        return jsonify({"error": "Insufficient balance."}), 400

    # Loyalty reward logic
    trip_count = user.get("count", 0) + 1
    loyalty_discount = 0
    reward_message = None

    if trip_count > 10:
        loyalty_discount = 5  # Discount after 10 trips
        fare = max(0, fare - loyalty_discount)  # Ensure fare is not negative
        reward_message = "Congratulations! You have earned a loyalty reward of ₹5."

    # Deduct fare, increment count, and update MongoDB
    new_balance = user["balance"] - fare
    collection.update_one({"aadhar": aadhar}, {"$set": {"balance": new_balance, "count": trip_count}})

   

    response = {
        "fare": fare,
        "new_balance": new_balance,
        "trip_count": trip_count,
    }
    if reward_message:
        response["reward_message"] = reward_message

    return jsonify(response), 200

@app.route('/view_user_details', methods=['GET'])
def get_user_details():
    aadhar = request.args.get('aadhar')

    if not aadhar:
        return render_template('view_user_details.html', error="Aadhar number is required.")

    # Fetch the user from the database
    user = collection.find_one({"aadhar": aadhar})

    if not user:
        return render_template('view_user_details.html', error="User not found. Please check the Aadhar number.")

    # Fetch the trip count
    trip_count = user.get("count", 0)

    return render_template('view_user_details.html', user=user, trip_count=trip_count)



if __name__ == "__main__":
    app.run(debug=True)
