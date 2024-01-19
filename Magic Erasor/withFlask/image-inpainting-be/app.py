from flask import Flask, Response, request , jsonify
from flask_cors import CORS

from serve.worker.deblur_worker import DeBlurWorker
from serve.worker.lama_worker import LaMaWorker
import io 

app = Flask(__name__)
CORS(app, supports_credentials=True)

lama_worker = LaMaWorker()
deblur_worker = DeBlurWorker()

@app.route('/api/effects/image/inpaint', methods=['POST'])
def inpaint():
    if request.method == 'POST':
        
        ## Working Program 
        files = request.files
        input_image = files['input_image'].read()
        input_mask = files['mask_image'].read()
        inpaint_image = lama_worker.process(input_image, input_mask)
        
        # print("inpaint_image :",inpaint_image[0])
        # print("inpaint_image Type :",type(inpaint_image[0]))
        
        return Response(inpaint_image[0], mimetype="image/jpeg")
        # return jsonify({"image": str(inpaint_image[0])}),200
  
        


        ## Testing Program 
        # files = request.files
        # input_image = files['origin'].read()
        # input_mask = files['mask'].read()

        # # Assuming lama_worker.process() returns a list of PIL Images
        # inpaint_images = lama_worker.process(input_image, input_mask)

        # # Take the first bytes from the list (modify as needed based on your use case)
        # inpaint_image_bytes = inpaint_images[0]

        # # Save the inpaint_image_bytes to a file
        # with open('inpaint_result.jpg', 'wb') as file:
        #     file.write(inpaint_image_bytes)

        # return Response(inpaint_image_bytes, mimetype="image/jpeg")
    
@app.route('/deblur', methods=['POST'])
def deblur():
    if request.method == 'POST':
        file = request.files['origin']
        input_image = file.read()
        deblur_image = deblur_worker.process(input_image)
        return Response(deblur_image, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=5000 , debug=True )