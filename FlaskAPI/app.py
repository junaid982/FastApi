from flask import Flask,jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({'msg':"Working "}),200




if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5000 , debug=True)



# FlaskApi/
#     |---Image_To_Pdf_1/
#     |       |---original_images/
#     |       |---resized_images/
#     |       |---pdf_files/
#     |---app.py


