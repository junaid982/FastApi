import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
from flask import Flask, request, jsonify
import torch
from model import WhiteBox
import easyocr as eo


app = Flask(__name__)


if not firebase_admin._apps:
    cred = credentials.Certificate("key.json")
    firebase_admin.initialize_app(cred, {
    'storageBucket': 'cartoonistic-toon-yourself.appspot.com'
    })

net = WhiteBox()
net.load_state_dict(torch.load("cartoon_model.pth"))
net.eval()
if torch.cuda.is_available():
    net.cuda()    


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to cartoonistic-toon-yourself-api!</h1>"

def cartoon1(img):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    return result.reshape(img.shape)

#     # Display sketch
#     cv2.imshow('cartoon image',cartoon)


#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# Nearly All Effects API Accepts firebase uid (for unique file name of that user) & Multi Part File with 'image' key

@app.route("/effects/sketch", methods=["POST"])
def img_to_sketch():
    uid = request.values.get('uid', None) 
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    k_size = 101

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert to Grey Image
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert Image
    invert_img=cv2.bitwise_not(grey_img)

    # Blur image
    blur_img=cv2.GaussianBlur(invert_img, (k_size,k_size),0)

    # Invert Blurred Image
    invblur_img=cv2.bitwise_not(blur_img)

    # Sketch Image
    sketch_img=cv2.divide(grey_img,invblur_img, scale=256.0)    

    
    _, buffer = cv2.imencode('.jpg', sketch_img)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})

@app.route("/effects/oilPaint", methods=["POST"])
def img_to_oil_paint():
    uid = request.values.get('uid', None) 
    file = request.files.get('images', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    output = cv2.xphoto.oilPainting(img, 7, 9)

    _, buffer = cv2.imencode('.jpg', output)
    
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})


@app.route("/effects/edge", methods=["POST"])
def img_to_edge():
    uid = request.values.get('uid', None) 
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=60, threshold2=100) 

    _, buffer = cv2.imencode('.jpg', edges)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})
    
@app.route("/effects/cartoon", methods=["POST"])  
def img_to_cartoon():

    uid = request.values.get('uid', None)
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

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

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    out = process(image)

    _, buffer = cv2.imencode('.jpg', out)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})

# Cartoon effect
@app.route('/effects/cartoon1', methods=["POST"])
def img_to_cartoon1():
    uid = request.values.get('uid', None) 
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = cartoon1(img=img)

    _, buffer = cv2.imencode('.jpg', result)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})

# Edge1 effect
@app.route('/effects/edge1', methods=["POST"])
def img_to_edge1():
    uid = request.values.get('uid', None) 
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

    _, buffer = cv2.imencode('.jpg', edges)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})

# Cartoon2 effect
@app.route('/effects/cartoon2', methods=["POST"])
def img_to_cartoon2():
    uid = request.values.get('uid', None) 
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = cartoon1(img=img)
    blurred = cv2.medianBlur(result, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    _, buffer = cv2.imencode('.jpg', cartoon)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})

# Marker effect
@app.route('/effects/marker', methods=["POST"])
def img_to_marker():
    uid = request.values.get('uid', None) 
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_style = cv2.stylization(img , sigma_s=150 , sigma_r=0.25)

    _, buffer = cv2.imencode('.jpg', img_style)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})

# Blur effect
@app.route('/effects/blur', methods=["POST"])
def img_to_edge2():
    uid = request.values.get('uid', None) 
    file = request.files.get('image', None)

    if(uid == None or file == None):
        return jsonify({'message' : 'uid & image are required field'})

    # Storage bucket firebase 
    bucket = storage.bucket()
    
    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # img_grey = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img,11)
    output = cv2.bilateralFilter(img_blur,15,75,75)

    _, buffer = cv2.imencode('.jpg', output)
    blob = bucket.blob(f'temp_images/{uid}.jpg')
    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')

    blob = bucket.get_blob(f'temp_images/{uid}.jpg')
    blob.make_public()

    # returns image url from firebase storage of modified image
    return jsonify({'image_url': blob.public_url})

# Image to text
@app.route('/imageToText', methods=["POST"])
def img_to_text():
    file = request.files.get('image', None)

    if(file == None):
        return jsonify({'message' : 'image is required field'})

    # Read Image
    file_bytes = np.fromfile(file, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    reader = eo.Reader(['en'])
    # reader = eo.Reader(['en'],gpu = False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    r_easy_ocr= reader.readtext(thresh,detail=0)


    # returns image url from firebase storage of modified image
    return jsonify({'result': "\n".join(r_easy_ocr)})


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5001)