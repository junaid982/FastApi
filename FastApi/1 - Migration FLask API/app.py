from flask import Flask , request , jsonify , send_file
from werkzeug.utils import secure_filename
import socket
from PIL import Image
import numpy as np
import cv2
import os 
import time
import threading



app  = Flask(__name__)

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)


# server_url = f"http:192.68.1.46:5000"
server_url = f"{IPAddr}:5000"

# Directories

# ================ 1 - Image to Sketch Folder Start =================

sketch_original_image_dir = "1_Sketch_Images/1_Original_Images"
sketch_final_image_dir =  "1_Sketch_Images/2_Final_Images"

os.makedirs(sketch_original_image_dir, exist_ok=True)
os.makedirs(sketch_final_image_dir, exist_ok=True)

# ================ 1 -  Image to Sketch Folder End  =================



# ================ 1 - Image to Sketch Program Start =================

def delete_Sketch_path(sketch_images):
    print("Delete running")
    for image in sketch_images:
        if os.path.exists(image):
            os.remove(image)
            print(f"Deleted: {image}")
        else:
            print(f"File not found: {image}")


@app.route("/api/effects/sketch/",methods=["POST"])
def sketch_images():
    try:
        
        images = request.files.getlist("images")
        
        sketch_images = []
    
        file_names = []
        
        k_size = 101
        
        
        if not images:
            return {"error": "Provided image to convert to sketch."}, 400
        
        if len(images) > 5:
            return {"error": "Upload max 5 images."}, 400
            
        
        for image in images:
            
            # Use the filename as the UID
            uid = image.filename
            file_names.append(uid)
            
            # Read the image
            # file_bytes = np.fromstring(image.read(), np.uint8)
            file_bytes = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Convert to Grey Image
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Invert Image
            invert_img = cv2.bitwise_not(grey_img)
            
            # Blur image
            blur_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)
            
            # Invert Blurred Image
            invblur_img = cv2.bitwise_not(blur_img)
            
            # Sketch Image
            sketch_img = cv2.divide(grey_img, invblur_img, scale=230.0)
            
            # Encode the sketch image to JPG
            _, buffer = cv2.imencode('.jpg', sketch_img)
            
            original_image_path = os.path.join(sketch_original_image_dir,uid)
            
            sketch_image_path = os.path.join(sketch_final_image_dir,uid)
            sketch_images.append(sketch_image_path)
            
            
            with open(original_image_path, 'wb') as original_image_file:
                original_image_file.write(file_bytes)
                
            with open(sketch_image_path, 'wb') as sketch_image_file:
                sketch_image_file.write(buffer.tobytes())
                
            # remove original uploaded image 
            os.remove(original_image_path)
            
        links = []
        for file_name in file_names:
            links.append(f"{server_url}/api/download/sketch/images/{file_name}" ) 
        
        threading.Timer(120, delete_Sketch_path, args=(sketch_images,)).start()

        
        return jsonify({"download_link":links}),200    
    
    
    except Exception as e:
        return jsonify({"error":str(e)}),500
    
        
@app.route("/api/download/sketch/images/<file_name>" , methods=["GET"])    
def download_sketch_images(file_name):
    try:
        print("file_name :",file_name)
        sketch_path = os.path.join(sketch_final_image_dir, file_name)
        if os.path.exists(sketch_path):
            return send_file(sketch_path, as_attachment=True, download_name=file_name)
        
        else:
            return jsonify({"error":"Image not Found"}),404
    
    except  Exception as e:
        return jsonify({"error":str(e)}),500


# ================ 1 -  Image to Sketch Program End  =================


if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5000 , debug=True , use_reloader=False)
    
