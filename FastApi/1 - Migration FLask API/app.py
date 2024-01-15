from flask import Flask , request , jsonify , send_file
from werkzeug.utils import secure_filename
import os 
import socket
import threading  

import asyncio

# Sketch Image
import numpy as np
import cv2

# Cartoon Image 
import torch
from cartoon_module.model import WhiteBox

from PIL import Image
import time



app  = Flask(__name__)

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

delete_path_timer = 20 

# server_url = f"http:192.68.1.46:5000"
server_url = f"{IPAddr}:5000"



#======================================= Directories Start From Here =======================================

# ================ 1 - Image to Sketch Folder Start =================

sketch_original_image_dir = "1_Sketch_Images/1_Original_Images"
sketch_final_image_dir =  "1_Sketch_Images/2_Final_Images"

os.makedirs(sketch_original_image_dir, exist_ok=True)
os.makedirs(sketch_final_image_dir, exist_ok=True)

# ================ 1 -  Image to Sketch Folder End  =================


# ================ 2 - Image to Sketch Folder Start =================


oilpaint_final_image_dir =  "2_OilPaint_Images/1_Final_Images"
os.makedirs(oilpaint_final_image_dir, exist_ok=True)

# ================ 2 -  Image to Sketch Folder End  =================


# ================ 3 - Image to Cartoon Folder Start =================

cartoon_final_image_dir =  "3_Cartoon_Images/1_Final_Images"
os.makedirs(cartoon_final_image_dir, exist_ok=True)

# ================ 3 - Image to Cartoon Folder ENd =================


#======================================= Directories End Here =======================================




# ======================================= Program Start From Here ======================================= 

# ================ 1 - Image to Sketch Program Start =================

def delete_Sketch_path(sketch_images):
    print("Delete delete_Sketch_path running")
    for image in sketch_images:
        if os.path.exists(image):
            os.remove(image)
            # print(f"Deleted: {image}")
        else:
            # print(f"File not found: {image}")
            pass


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
            sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)
            
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
        
        threading.Timer(delete_path_timer, delete_Sketch_path, args=(sketch_images,)).start()
    
        # download_sketch_images(sketch_images)
        
        return jsonify({"message":"Download Link Available only for 10 minuts","download_link":links}),200  
    
    
    except Exception as e:
        return jsonify({"error":str(e)}),500
    
        
@app.route("/api/download/sketch/images/<file_name>" , methods=["GET"])    
def download_sketch_images(file_name):
    try:
        # print("file_name :",file_name)
        sketch_path = os.path.join(sketch_final_image_dir, file_name)
        if os.path.exists(sketch_path):
            return send_file(sketch_path, as_attachment=True, download_name=file_name)
        
        else:
            return jsonify({"error":"Image not Found"}),404
    
    except  Exception as e:
        return jsonify({"error":str(e)}),500


# ================ 1 -  Image to Sketch Program End  =================

# ================ 2 -  Image to OilPainting Program Start From Here  =================

def delete_OilPaint_path(sketch_images):
    time.sleep(delete_path_timer)
    print("Delete delete_OilPaint_path running")
    for image in sketch_images:
        # print("image",image)
        if os.path.exists(image):
            os.remove(image)
            print(f"Deleted: {image}")
        else:
            pass
            # print(f"File not found: {image}")
                


@app.route("/api/effects/oil-paint/", methods=["POST"])
def oilpainting_image():
    try:
        
        images = request.files.getlist("images")
        size = int(request.form.get("size", 6))  # Default size: 8
        dynRatio = int(request.form.get("dynRatio", 8))  # Default dynRatio: 10

        oilpainting_images = []
        file_names = []

        if not images:
            return {"error": "Provided image to convert to oil paint."}, 400
        
        if len(images) > 5:
            return {"error": "Upload max 5 images."}, 400
        
        if size < 0 and size > 15:
            return {"error": "size paramater should be in between 1 to 15."}, 400
        
        if dynRatio < 0 and dynRatio > 15:
            return {"error": "dynRatio paramater should be in between 1 to 15."}, 400
        

        for image in images:
            uid = image.filename

            file_names.append(uid)

            # Read Image
            file_bytes = np.fromfile(image, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # output = cv2.xphoto.oilPainting(img, 7, 9)
            output = cv2.xphoto.oilPainting(img, size, dynRatio)

            _, buffer = cv2.imencode('.jpg', output)

            oilpaint_final_path = os.path.join(oilpaint_final_image_dir , uid)
            oilpainting_images.append(oilpaint_final_path)

            # Store converted Images
            with open(oilpaint_final_path , 'wb') as oilpaint_image_file:
                oilpaint_image_file.write(buffer.tobytes())


        links = []
        for file_name in file_names:
            links.append(f"{server_url}/api/download/oil/paint/image/{file_name}" ) 
        
        threading.Timer(delete_path_timer, delete_OilPaint_path, args=(oilpainting_images,)).start()
        
        return jsonify({"message":"Download Link Available only for 10 minuts","download_link":links}),200  

    except  Exception as e:
        return jsonify({"error":str(e)}),500
    


@app.route("/api/download/oil/paint/image/<file_name>" , methods=["GET"])    
def download_oilpainting_images(file_name):
    try:
        # print("file_name :",file_name)
        oilpaint_path = os.path.join(oilpaint_final_image_dir, file_name)
        if os.path.exists(oilpaint_path):
            return send_file(oilpaint_path, as_attachment=True, download_name=file_name)
        
        else:
            return jsonify({"error":"Image not Found"}),404
    
    except  Exception as e:
        return jsonify({"error":str(e)}),500




# ================ 2 -  Image to OilPainting Program End Here =================


# ================ 3 -  Image to Cartoon Program Start From Here  =================

net = WhiteBox()
net.load_state_dict(torch.load(r"cartoon_module/cartoon_model.pth"))
net.eval()
if torch.cuda.is_available():
    net.cuda()    


def delete_Cartoon_path(sketch_images):
    print("Delete delete_Cartoon_path running")
    for image in sketch_images:
        if os.path.exists(image):
            os.remove(image)
            # print(f"Deleted: {image}")
        else:
            # print(f"File not found: {image}")
            pass


@app.route("/api/effects/cartoon/", methods=["POST"])
def cartoon_image():
    try:
        
        images = request.files.getlist("images")
        # print("images :",type(images[0]))

        cartoon_images = []
        file_names = []

        if not images:
            return {"error": "Provided image to convert to oil paint."}, 400
        
        for image in images:
            # getting file name and store into a list 
            uid = image.filename
            file_names.append(uid)

            def resize_crop(image):
                h, w, _ = np.shape(image)
                if min(h, w) > 720:
                    if h > w:
                        h, w = int(720*h/w), 720
                    else:
                        h, w = 720, int(720*w/h)
                image = cv2.resize(image, (w, h),
                                interpolation=cv2.INTER_AREA)
                h, w = (h//8)*8, (w//8)*8
                image = image[:h, :w, :]
                return image
            
            def process(img):
                img = resize_crop(img)
                img = img.astype(np.float32)/127.5 - 1
                img_tensor = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                with torch.no_grad():
                    output = net(img_tensor, r=1, eps=5e-3)

                output = 127.5*(output.cpu().numpy().squeeze().transpose(1, 2, 0)+1)
                output = np.clip(output, 0, 255).astype(np.uint8)
                return output
            
            # Read Image
            file_bytes = np.fromfile(image, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            out = process(img)

            _, buffer = cv2.imencode('.jpg', out)

            cartoon_final_path = os.path.join(cartoon_final_image_dir , uid)
            cartoon_images.append(cartoon_final_path)

            # Store converted Images
            with open(cartoon_final_path , 'wb') as cartoon_image_file:
                cartoon_image_file.write(buffer.tobytes())

        links = []

        for file_name in file_names:
            links.append(f"{server_url}/api/download/cartoon/image/{file_name}" ) 

        threading.Timer(delete_path_timer , delete_Cartoon_path, args=(cartoon_images,)).start()
        
        return jsonify({"message":"Download Link Available only for 10 minuts","image_url":links}),200  

    except  Exception as e:
        return jsonify({"error":str(e)}),500


@app.route("/api/download/cartoon/image/<file_name>" , methods=["GET"])    
def download_Cartoon_path(file_name):
    try:
        # print("file_name :",file_name)
        cartoon_path = os.path.join(cartoon_final_image_dir, file_name)
        if os.path.exists(cartoon_path):
            return send_file(cartoon_path, as_attachment=True, download_name=file_name)
        
        else:
            return jsonify({"error":"Image not Found"}),404
    
    except  Exception as e:
        return jsonify({"error":str(e)}),500






# ================ 3 -  Image to Cartoon Program End Here =================



# ======================================= Program End Here ======================================= 

if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5000 , debug=True , use_reloader=False)
    