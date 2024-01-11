from flask import Flask,jsonify,request
from werkzeug.utils import secure_filename

import os 


app = Flask(__name__)

server_url = "http://192.168.0.104:5000"

@app.route('/')


def home():
    return jsonify({'msg':"Working "}),200


# ========= 1 - Convert Image to pdf =========
desired_width = 600
desired_height = 700

# Directories
image_files_original = "Image_To_Pdf_1/original_images"
image_to_pdf_resized = "Image_To_Pdf_1/resized_images"
image_to_pdf_file = "Image_To_Pdf_1/pdf_files"

# Create directories if they don't exist
os.makedirs(image_files_original, exist_ok=True)
os.makedirs(image_to_pdf_resized, exist_ok=True)
os.makedirs(image_to_pdf_file, exist_ok=True)


@app.route("/api/converter/img/to/pdf/",methods=["POST"])
def convert_to_pdf():
    try:
        uploaded_image_paths = []
        resized_image_path = []
        file_name = ''

        images = request.files.getlist("images")
        for image in images:
            original_image_path = os.path.join(image_files_original , secure_filename(image.filename))
            resized_image_path = os.path.join(image_to_pdf_resized, secure_filename(image.filename))

            file_name = file_name + secure_filename(image.filename).split('.')[0]

            # Save the uploaded image to the original directory
            image.save(original_image_path)


            


        
        return jsonify("images[0]"),200
    
    except Exception as e:
        return jsonify(str(e)),500

if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5000 , debug=True)



# FlaskApi/
#     |---Image_To_Pdf_1/
#     |       |---original_images/
#     |       |---resized_images/
#     |       |---pdf_files/
#     |---app.py


