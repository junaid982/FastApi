
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse ,JSONResponse
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
# image to pdf 
from typing import List
import asyncio


import os

import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
# from flask import Flask, request, jsonify
import torch
from model import WhiteBox
# import easyocr as eo

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path
from PIL import Image, ImageFilter




app = FastAPI(root_path="/myapi")

server_url = "http://127.0.0.1:8000"

# =================================== Blure Images Directory Start

# Create a directory to store uploaded images
blur_image_original = 'blur_image_original'
os.makedirs(blur_image_original, exist_ok=True)

blur_image_final = 'blur_image_final'
os.makedirs(blur_image_final, exist_ok=True)

# =================================== Blure Images Directory End



# =================================== Sketch Images Directory Start

# Define the directories to store original and sketch images
original_image_dir = "sketch_image_original"
final_image_dir = "sketch_image_final"

# Create the directories if they don't exist
Path(original_image_dir).mkdir(parents=True, exist_ok=True)
Path(final_image_dir).mkdir(parents=True, exist_ok=True)


# =================================== Sketch Images Directory End

# =================================== Edge Images Directory Start

# Define the directories to store original and edge images
edge_original = "edge_original"
edge_final = "edge_final"

# Create the directories if they don't exist
Path(edge_original).mkdir(parents=True, exist_ok=True)
Path(edge_final).mkdir(parents=True, exist_ok=True)

# =================================== Edge Images Directory End


# =================================== Cartoon Images Directory Start

# Load the WhiteBox model and move it to GPU if available
net = WhiteBox()
net.load_state_dict(torch.load("cartoon_model.pth"))
net.eval()
if torch.cuda.is_available():
    net.cuda()

# Define the directories to store original and cartoon images
cartoon_original_dir = "cartoon_original"
cartoon_final_dir = "cartoon_final"

# Create the directories if they don't exist
Path(cartoon_original_dir).mkdir(parents=True, exist_ok=True)
Path(cartoon_final_dir).mkdir(parents=True, exist_ok=True)



# =================================== Cartoon Images Directory End


# =================================== Oil painting Images Directory Start


# Define the directories to store original and oil painting images
oil_painting_original_dir = "oil_painting_original"
oil_painting_final_dir = "oil_painting_final"

# Create the directories if they don't exist
# Path(original_image_dir).mkdir(parents=True, exist_ok=True)
# Path(final_image_dir).mkdir(parents=True, exist_ok=True)

Path(oil_painting_original_dir).mkdir(parents=True, exist_ok=True)
Path(oil_painting_final_dir).mkdir(parents=True, exist_ok=True)

# =================================== Oil painting Images Directory End




@app.post('/')
def read_root():
    # raise HTTPException(status_code=404, detail="File not found")
    # FileResponse(docx_path, headers={"Content-Disposition": f"attachment; filename={docx_file_name}"})
    return {"404": "Server not working!"}


@app.post("/api/effects/blur/")
async def img_to_sketch(images: List[UploadFile] = File(...) ,background_tasks: BackgroundTasks = BackgroundTasks()):

    try:
        k_size = 101

        original_image_paths = []
        blurred_image_paths = []
        download_urls = []

        for image in images:
            file_name = image.filename
            original_image_path = os.path.join(blur_image_original, file_name )
            final_image_dir = os.path.join(blur_image_final, file_name )

            # Save the original image to the "blur_image_original" folder.
            with open( original_image_path , "wb") as image_file:
                image_file.write(image.file.read())

            # Open the image for blurring.
            original_image = Image.open(original_image_path)

            # Apply a Gaussian blur to the image.
            blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius=10))

            # Save the blurred image to the "blur_image_final" folder.
            blurred_image.save(final_image_dir)

            # Provide the download URL for the blurred image.

            original_image_paths.append(original_image_path)

            download_url = f"{server_url}/download/blur-image/{file_name}"
            blurred_image_paths.append(final_image_dir)

            download_urls.append(download_url)

        # print('blurred_image_paths :',blurred_image_paths)

        background_tasks.add_task(delete_blur_path, original_image_paths , blurred_image_paths )

        return { "message":"Download link availabel only for 5 min.","download_url": download_urls}

    except:
        return JSONResponse(content={'message': 'Server Error '}, status_code=400)



@app.get("/download/blur-image/{image_name}")
async def download_image(image_name: str):
    try:

        # Define the path to the blurred image.
        blurred_image_path = f"{blur_image_final}/{image_name}"
        if not os.path.exists(blurred_image_path):

            raise HTTPException(status_code=404, detail="Link Expired")
        
        # Return the blurred image as a downloadable response.
        return FileResponse(blurred_image_path)
    except:
        return JSONResponse(content={'message': 'Error while downloading maybe Link Expired'}, status_code=400)


async def delete_blur_path(original_image_paths , blurred_image_paths ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    for original_path in original_image_paths:
        if os.path.exists(original_path):
            os.remove(original_path)

    for blur_path in blurred_image_paths:
        if os.path.exists(blur_path):
            os.remove(blur_path)


    


## =================================== Sketch Images  Start

@app.post("/api/effects/sketch/")
async def img_to_sketch(images: List[UploadFile] = File(...) ,background_tasks: BackgroundTasks = BackgroundTasks()):
    try:

        if not images:
            return JSONResponse(content={'message': 'image is a required field'}, status_code=400)

        k_size = 101

        original_image_paths = []
        sketch_image_paths = []
        download_image_paths = []

        for image in images :
            # Use the filename as the UID
            uid = image.filename

            # Read the image
            file_bytes = np.fromstring(await image.read(), np.uint8)
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

            # Save the original and sketch images to directories
            original_image_path = f"{original_image_dir}/{uid}"
            sketch_image_path = f"{final_image_dir}/{uid}"

            with open(original_image_path, 'wb') as original_image_file:
                original_image_file.write(file_bytes)

            with open(sketch_image_path, 'wb') as sketch_image_file:
                sketch_image_file.write(buffer.tobytes())


            print('sketch_image_path :',sketch_image_path)
            # Return the local file path for the sketch image
        

            download_url = f"{server_url}/download/sketch-image/{uid}"

            original_image_paths.append(original_image_path)
            sketch_image_paths.append(sketch_image_path)
            download_image_paths.append(download_url)

        # print('original_image_paths :',original_image_paths)
        # print('sketch_image_paths :',sketch_image_paths)

        background_tasks.add_task(delete_sketch_path, original_image_paths , sketch_image_paths )

        return JSONResponse(content={"message":"Download link availabel only for 5 min.",'image_url': download_image_paths})

    except:
        return JSONResponse(content={'message': 'Server Error .'}, status_code=400)



@app.get("/download/sketch-image/{image_name}")
async def download_image( image_name: str):
    try:

        # Define the path to the blurred image.
        sketch_image_path = f"{final_image_dir}/{image_name}"
    
        if not os.path.exists(sketch_image_path):

            raise HTTPException(status_code=404, detail="Link Expired")
        
        # Return the blurred image as a downloadable response.
        return FileResponse(sketch_image_path)
    except :
        return JSONResponse(content={'message': 'Error while downloading.'}, status_code=400)


async def delete_sketch_path(original_image_paths , sketch_image_paths ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    for original_path in original_image_paths:
        if os.path.exists(original_path):
            os.remove(original_path)

    for sketch in sketch_image_paths:
        if os.path.exists(sketch):
            os.remove(sketch)



## =================================== Stech Images  End



# =================================== Edge Images Start


@app.post("/api/effects/edge/")
async def img_to_edge(images: List[UploadFile] = File(...) ,background_tasks: BackgroundTasks = BackgroundTasks()):

    if not images:
        return JSONResponse(content={'message': 'image is a required field'}, status_code=400)

    try:


        edge_original_paths = []
        edge_image_paths = []
        download_urls = []
        
        for image in images:

            # Use the filename as the UID
            uid = image.filename

            # Read the image
            file_bytes = np.fromstring(await image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Convert the image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

            # Canny Edge Detection
            edges = cv2.Canny(image=img_blur, threshold1=60, threshold2=100)

            # Encode the edge image to JPG
            _, buffer = cv2.imencode('.jpg', edges)

            # Save the original and edge images to directories
            original_image_path = f"{edge_original}/{uid}"
            edge_original_path = f"{edge_original}/{uid}"
            edge_image_path = f"{edge_final}/{uid}"

            with open(edge_original_path, 'wb') as original_image_file:
                original_image_file.write(file_bytes)

            with open(edge_image_path, 'wb') as edge_image_file:
                edge_image_file.write(buffer.tobytes())

            print('edge_original_path :',edge_original_path)
            print('edge_image_path :',edge_image_path)

            download_url = f"{server_url}/download/edge-image/{uid}"

            edge_original_paths.append(edge_original_path)
            edge_image_paths.append(edge_image_path)
            download_urls.append(download_url)

        background_tasks.add_task(delete_edge_path, edge_original_paths , edge_image_paths )

        # Return the local file path for the edge image
        return JSONResponse(content={"message":"Download link availabel only for 5 min.",'image_url': download_urls})

    except:
        return JSONResponse(content={'message': 'Server Error '}, status_code=400)





@app.get("/download/edge-image/{image_name}")
async def download_image(image_name: str):
    try:

        # Define the path to the blurred image.
        edge_image_path = f"{edge_final}/{image_name}"
    
        if not os.path.exists(edge_image_path):

            raise HTTPException(status_code=404, detail="Link Expired")
        
        # Return the blurred image as a downloadable response.
        return FileResponse(edge_image_path)
    except :
        return JSONResponse(content={'message': 'Error while downloading.'}, status_code=400)


async def delete_edge_path(edge_original_paths , edge_image_paths):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    for original_path in edge_original_paths:
        if os.path.exists(original_path):
            os.remove(original_path)

    for edge in edge_image_paths:
        if os.path.exists(edge):
            os.remove(edge)





# =================================== Edge Images End



# =================================== Cartoon Images Start

@app.post("/api/effects/cartoon/")
async def img_to_cartoon(images: List[UploadFile] = File(...) ,background_tasks: BackgroundTasks = BackgroundTasks()):

    if not images:
        return JSONResponse(content={'message': 'image is a required field'}, status_code=400)

    try:

        cartoon_original_paths = []
        cartoon_image_paths = []
        download_urls = []

        for image in images:


            # Use the filename as the UID
            uid = image.filename

            def resize_crop(image):
                h, w, _ = np.shape(image)
                if min(h, w) > 720:
                    if h > w:
                        h, w = int(720 * h / w), 720
                    else:
                        h, w = 720, int(720 * w / h)
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                h, w = (h // 8) * 8, (w // 8) * 8
                image = image[:h, :w, :]
                return image

            def process(img):
                img = resize_crop(img)
                img = img.astype(np.float32) / 127.5 - 1
                img_tensor = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                with torch.no_grad():
                    output = net(img_tensor, r=1, eps=5e-3)
                output = 127.5 * (output.cpu().numpy().squeeze().transpose(1, 2, 0) + 1)
                output = np.clip(output, 0, 255).astype(np.uint8)
                return output

            # Read Image
            file_bytes = np.fromstring(await image.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            out = process(image)

            # Save the original and cartoon images to directories
            cartoon_original_path = f"{cartoon_original_dir}/{uid}"
            cartoon_final_path = f"{cartoon_final_dir}/{uid}"

            with open(cartoon_original_path, 'wb') as original_image_file:
                original_image_file.write(file_bytes)

            cv2.imwrite(cartoon_final_path, out)

            download_url = f"{server_url}/download/cartoon-image1/{uid}"

            cartoon_original_paths.append(cartoon_original_path)
            cartoon_image_paths.append(cartoon_final_path)
            download_urls.append(download_url)


        background_tasks.add_task(delete_cartoon_path, cartoon_original_paths , cartoon_image_paths )

        # Return the local file path for the cartoon image
        return JSONResponse(content={"message":"Download link availabel only for 5 min.",'image_url': download_urls})

    except:
        return JSONResponse(content={'message': 'Server Error '}, status_code=400)




@app.get("/download/cartoon-image1/{image_name}")
async def download_image(image_name: str):
    try:

        # Define the path to the blurred image.
        cartoon_image_path = f"{cartoon_final_dir}/{image_name}"
        print('cartoon_image_path download :',cartoon_image_path)
        if not os.path.exists(cartoon_image_path):

            raise HTTPException(status_code=404, detail="Link Expired")
        
        # Return the blurred image as a downloadable response.
        return FileResponse(cartoon_image_path)
    except :
        return JSONResponse(content={'message': 'Error while downloading.'}, status_code=400)


async def delete_cartoon_path(cartoon_original_paths , cartoon_image_paths ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    for original_path in cartoon_original_paths:
        if os.path.exists(original_path):
            os.remove(original_path)

    for cartoon_path in cartoon_image_paths:
        if os.path.exists(cartoon_path):
            os.remove(cartoon_path)


# =================================== Cartoon Images ENd



# =================================== Oil painting Images  Start




def oil_painting_effect(image, radius, intensity):
    output = cv2.stylization(image, sigma_s=radius, sigma_r=intensity)
    return output

@app.post("/api/effects/oil-paint-control/")
async def img_to_oil_paint(image: UploadFile = File(...),radius: int = Form(...),intensity: int = Form(...) ,background_tasks: BackgroundTasks = BackgroundTasks()):
# async def img_to_oil_paint(image: UploadFile, radius: int = 7, intensity: int = 9):
    if not image:
        return JSONResponse(content={'message': 'image is a required field'}, status_code=400)

    try:

        # Use the filename as the UID
        uid = image.filename

        # Read Image
        file_bytes = np.fromstring(await image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Apply the custom oil painting effect
        output = oil_painting_effect(img, radius, intensity)

        # Save the original and oil painting images to directories
        oil_painting_original_path = f"{oil_painting_original_dir}/{uid}"
        oil_painting_final_path = f"{oil_painting_final_dir}/{uid}"

        with open(oil_painting_original_path, 'wb') as original_image_file:
            original_image_file.write(file_bytes)

        cv2.imwrite(oil_painting_final_path, output)

        download_url = f"{server_url}/download/oil-painting-control/{uid}"

        background_tasks.add_task(delete_oil_painting_control_path, oil_painting_original_path , oil_painting_final_path )


        # Return the local file path for the oil painting image
        return JSONResponse(content={"message":"Download link availabel only for 5 min.",'image_url': download_url})

    except:
        return JSONResponse(content={'message': 'Server Error '}, status_code=400)




@app.get("/download/oil-painting-control/{image_name}")
async def download_image( image_name: str):

    # try:

        # Define the path to the blurred image.
        oil_painting_control_path = f"{oil_painting_final_dir}/{image_name}"

        if not os.path.exists(oil_painting_control_path):

            raise HTTPException(status_code=404, detail="Link Expired")
        
        # Return the blurred image as a downloadable response.
        return FileResponse(oil_painting_control_path)
    # except:
    #     return JSONResponse(content={'message': 'Error while downloading maybe Link Expired'}, status_code=400)


async def delete_oil_painting_control_path(oil_painting_original_path , oil_painting_final_path ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    print('oil_painting_original_path :',oil_painting_original_path)
    print('oil_painting_final_path :',oil_painting_final_path)

    if os.path.exists(oil_painting_original_path):
        os.remove(oil_painting_original_path)


    if os.path.exists(oil_painting_final_path):
        os.remove(oil_painting_final_path)







# =================================== Oil painting Images End



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
