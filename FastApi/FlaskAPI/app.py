# from flask import Flask,jsonify,request,send_file
# from werkzeug.utils import secure_filename
# from PIL import Image
# import os 
# import img2pdf
# import asyncio
# import threading


# app = Flask(__name__)

# server_url = "http://192.168.0.104:5000"

# @app.route('/')
# def home():
#     print("request.data : ",request.form.get("key"))
#     print("request.data type : ",type(request.form.get("key")))

#     return jsonify({'msg':"Working "}),200


# # ========= 1 - Convert Image to pdf =========
# desired_height = 1700
# desired_width = 600

# # Directories
# image_files_original = "Image_To_Pdf_1/original_images"
# image_to_pdf_resized = "Image_To_Pdf_1/resized_images"
# image_to_pdf_file = "Image_To_Pdf_1/pdf_files"

# # Create directories if they don't exist
# os.makedirs(image_files_original, exist_ok=True)
# os.makedirs(image_to_pdf_resized, exist_ok=True)
# os.makedirs(image_to_pdf_file, exist_ok=True)


# @app.route("/api/converter/img/to/pdf/",methods=["POST"])
# def convert_to_pdf():
#     try:
#         uploaded_image_paths = []
#         resized_image_paths = []
#         file_name = ''

#         images = request.files.getlist("images")
#         resized = request.form.get("resized" , None)
#         print("resized : ",request.form.get("resized" , None))
#         print("resized type : ",type(request.form.get("resized" ,None)))
            
#         if not images:
#             return {"error": "Provided image to convert pdf ."}, 400

#         for image in images:
#             original_image_path = os.path.join(image_files_original , secure_filename(image.filename))
#             resized_image_path = os.path.join(image_to_pdf_resized, secure_filename(image.filename))

#             file_name = file_name + secure_filename(image.filename).split('.')[0]

#             # Save the uploaded image to the original directory
#             image.save(original_image_path)

           
#             # Open the uploaded image using Pillow
            
#             with Image.open(original_image_path) as img:
#                 # Resize the image to the desired dimensions with the BILINEAR filter
#                 if resized == "true":
#                     img = img.resize((desired_width, desired_height), Image.BILINEAR)
            
#                 # Save the resized image to the resized directory
#                 img.save(resized_image_path)

#             resized_image_paths.append(resized_image_path)
            
#             uploaded_image_paths.append(original_image_path)

#         if resized == "true":
#             pdf_bytes = img2pdf.convert(resized_image_paths)
#         else:
#             pdf_bytes = img2pdf.convert(uploaded_image_paths)

#         file_name = file_name + ".pdf"

#         pdf_path = os.path.join(image_to_pdf_file, file_name)

#         with open(pdf_path, "wb") as pdf_file:
#             pdf_file.write(pdf_bytes)

#         for image_path in uploaded_image_paths:
#             os.remove(image_path)

#         for image_path in resized_image_paths:
#             os.remove(image_path)

#         delete_pdf_path(pdf_path)

#         download_url = f"{server_url}/download/pdf/{file_name}"


#         link =  {
#             "message": "Link valid only for 5 min.",
#             "Download url": download_url,
#             # "resized": resized,
#             # "width": width
#         }
        
#         # return jsonify(link),200
#         pdf_path = os.path.join(image_to_pdf_file, file_name)

#         if not os.path.exists(pdf_path):
#             app.logger.error(f"File not found: {pdf_path}")
#             return "Link Expired", 404
        
#         return send_file(pdf_path, as_attachment=True, download_name=file_name)

    
#     except Exception as e:
#         return jsonify({"error":str(e)}),500
    


# def delete_pdf_path(pdf_file_path):
#     print("Delete running")
#     threading.Timer(60, os.remove, args=[pdf_file_path]).start()


# @app.route("/download/pdf/<file_name>")
# def download_pdf(file_name):
#     pdf_path = os.path.join(image_to_pdf_file, file_name)

#     if not os.path.exists(pdf_path):
#         app.logger.error(f"File not found: {pdf_path}")
#         return "Link Expired", 404
    
#     return send_file(pdf_path, as_attachment=True, download_name=file_name)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0' , port=5000 , debug=True , use_reloader=False)



# # FlaskApi/
# #     |---Image_To_Pdf_1/
# #     |       |---original_images/
# #     |       |---resized_images/
# #     |       |---pdf_files/
# #     |---app.py




from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import os
import img2pdf
import threading

app = Flask(__name__)

server_url = "http://192.168.0.104:5000"
# ========= 1 - Convert Image to pdf =========
desired_height = 1700
desired_width = 600

# Directories
image_files_original = "Image_To_Pdf_1/original_images"
image_to_pdf_resized = "Image_To_Pdf_1/resized_images"
image_to_pdf_file = "Image_To_Pdf_1/pdf_files"

# Create directories if they don't exist
os.makedirs(image_files_original, exist_ok=True)
os.makedirs(image_to_pdf_resized, exist_ok=True)
os.makedirs(image_to_pdf_file, exist_ok=True)


@app.route('/')
def home():
    print("request.data : ", request.form.get("key"))
    print("request.data type : ", type(request.form.get("key")))

    return jsonify({'msg': "Working "}), 200

# ... (rest of your code)

@app.route("/api/converter/img/to/pdf/", methods=["POST"])
def convert_to_pdf():
    try:
        uploaded_image_paths = []
        resized_image_paths = []
        file_name = ''

        images = request.files.getlist("images")
        resized = request.form.get("resized", None)
        print("resized : ", request.form.get("resized", None))
        print("resized type : ", type(request.form.get("resized", None)))

        if not images:
            return {"error": "Provided image to convert pdf."}, 400

        for image in images:
            original_image_path = os.path.join(image_files_original, secure_filename(image.filename))
            resized_image_path = os.path.join(image_to_pdf_resized, secure_filename(image.filename))

            file_name = file_name + secure_filename(image.filename).split('.')[0]

            # Save the uploaded image to the original directory
            image.save(original_image_path)

            # Open the uploaded image using Pillow
            with Image.open(original_image_path) as img:
                # Resize the image to the desired dimensions with the BILINEAR filter
                if resized == "true":
                    img = img.resize((desired_width, desired_height), Image.BILINEAR)

                # Save the resized image to the resized directory
                img.save(resized_image_path)

            resized_image_paths.append(resized_image_path)
            uploaded_image_paths.append(original_image_path)

        if resized == "true":
            pdf_bytes = img2pdf.convert(resized_image_paths, rotation=img2pdf.Rotation.auto())
        else:
            pdf_bytes = img2pdf.convert(uploaded_image_paths, rotation=img2pdf.Rotation.auto())

        file_name = file_name + ".pdf"
        pdf_path = os.path.join(image_to_pdf_file, file_name)

        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)

        for image_path in uploaded_image_paths:
            os.remove(image_path)

        for image_path in resized_image_paths:
            os.remove(image_path)

        delete_pdf_path(pdf_path)

        download_url = f"{server_url}/download/pdf/{file_name}"

        link = {
            "message": "Link valid only for 5 min.",
            "Download url": download_url,
        }

        pdf_path = os.path.join(image_to_pdf_file, file_name)

        if not os.path.exists(pdf_path):
            app.logger.error(f"File not found: {pdf_path}")
            return "Link Expired", 404

        return send_file(pdf_path, as_attachment=True, download_name=file_name)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def delete_pdf_path(pdf_file_path):
    print("Delete running")
    threading.Timer(60, os.remove, args=[pdf_file_path]).start()


@app.route("/download/pdf/<file_name>")
def download_pdf(file_name):
    pdf_path = os.path.join(image_to_pdf_file, file_name)

    if not os.path.exists(pdf_path):
        app.logger.error(f"File not found: {pdf_path}")
        return "Link Expired", 404
    
    return send_file(pdf_path, as_attachment=True, download_name=file_name)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
