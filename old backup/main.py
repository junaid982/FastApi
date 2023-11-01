# Import required libraries
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from starlette.responses import FileResponse
from PIL import Image
import os
import img2pdf
from fastapi.staticfiles import StaticFiles

# image to pdf 
from typing import List
# import fitz  # PyMuPDF

# doc to pdf 
# from pydantic import BaseModel
import os
from fastapi.responses import FileResponse
from docx2pdf import convert
import asyncio
import re
from pdf2docx import Converter


# remove background 
from rembg import remove
import shutil
import os
import subprocess

# text extraction from image
import easyocr


# text to audio
from gtts import gTTS
from pydub import AudioSegment
import io

app = FastAPI(root_path="/myapi")

server_url = "http://127.0.0.1:8000"


# ======================= 1 - Image to pdf Directories start =======================  
# os.makedirs('image_to_pdf', exist_ok=True)

# Create a directory to store uploaded images
image_files_original = 'image_files_original'
os.makedirs(image_files_original, exist_ok=True)

# Create a directory to store resized images
image_to_pdf_resized = 'image_to_pdf_resized'
os.makedirs(image_to_pdf_resized, exist_ok=True)

image_to_pdf_file = 'image_to_pdf_file'
os.makedirs(image_to_pdf_file ,exist_ok=True )

# ======================= 1 - Image to pdf Directories end =======================  


# ======================= 2 - Docx to pdf Directories start =======================  

# os.makedirs('docx_to_pdf', exist_ok=True)

# Create a directory to store uploaded images
docx_to_pdf_original = 'docx_to_pdf_original'
os.makedirs(docx_to_pdf_original, exist_ok=True)

# ======================= 2 - Docx to pdf Directories end =======================  


# ======================= 3 - pdf to doc Directories start =======================  

# Create a directory to store uploaded PDF files
pdf_to_docx_original = 'pdf_to_docx_original'
os.makedirs(pdf_to_docx_original, exist_ok=True)

# ======================= 3 - pdf to doc Directories end =======================  

# ======================= 4 - Remove Image Background Directory  start =======================

image_remove_bg_original = 'image_remove_bg_original' 
os.makedirs(image_remove_bg_original, exist_ok=True)

image_remove_bg_final = 'image_remove_bg_final'
os.makedirs(image_remove_bg_final, exist_ok=True)

# ======================= 4 - Remove Image Background Directory end =======================

# ======================= 5 - Replace background Image Directory start =======================

RemoveBackground_replace_bg = 'RemoveBackground'
os.makedirs(RemoveBackground_replace_bg, exist_ok=True)

ReplaceBackground = 'ReplaceBackground'
os.makedirs(ReplaceBackground, exist_ok=True)

# ======================= 5 - Replace background Image Directory end =======================



# ======================= All Mounted Media Folders Start =======================




# ======================= All Mounted Media Folders End =======================

app.mount("/media1", StaticFiles(directory="image_files_original"), name="media1")
app.mount("/media2", StaticFiles(directory="docx_to_pdf_original"), name="media2")
app.mount("/media3", StaticFiles(directory="pdf_to_docx_original"), name="media3")
app.mount("/media4", StaticFiles(directory="image_remove_bg_original"), name="media4")
app.mount("/media5", StaticFiles(directory="image_remove_bg_final"), name="media5")
app.mount("/media6", StaticFiles(directory="RemoveBackground"), name="media6")
app.mount("/media7", StaticFiles(directory="ReplaceBackground"), name="media7")

# ======================= 1- Image to pdf Directories Start  =======================  
# Define the desired image size
desired_width = 600
desired_height = 700


@app.get('/')
def read_root():
    return {"404": "Server not working!"}

@app.post("/convert/img/to/pdf/")
async def convert_to_pdf(images: List[UploadFile] = File(...) ,background_tasks: BackgroundTasks = BackgroundTasks()):
    uploaded_image_paths = []
    resized_image_paths = []
    
    file_name = ''
    for image in images:
        # Construct the file path to save the uploaded image
        original_image_path = os.path.join(image_files_original, image.filename)
        print('original_image_path :',original_image_path)
        
        resized_image_path = os.path.join(image_to_pdf_resized, image.filename)
        # print('resized_image_path :',resized_image_path)


        file_name = file_name + image.filename.split('.')[0]
        # print('file_name :',file_name)
        

        # Save the uploaded image to the original directory
        with open(original_image_path, "wb") as image_file:
            image_data = image.file.read()
            image_file.write(image_data)
        
        # Open the uploaded image using Pillow
        with Image.open(original_image_path) as img:
            # Resize the image to the desired dimensions with the BILINEAR filter
            img = img.resize((desired_width, desired_height), Image.BILINEAR)
            
            # Save the resized image to the resized directory
            img.save(resized_image_path)
        
        uploaded_image_paths.append(original_image_path)
        resized_image_paths.append(resized_image_path)

    pdf_bytes = img2pdf.convert(resized_image_paths)

    file_name = file_name + ".pdf"
    pdf_path = os.path.join(image_to_pdf_file , file_name)
    
    with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)


    for image_path in uploaded_image_paths:
            os.remove(image_path)

    for image_path in resized_image_paths:
            os.remove(image_path)

    background_tasks.add_task(delete_pdf_path, pdf_path )

    
    download_urls = f"{server_url}/download/pdf/{image_to_pdf_file}/{file_name}"


    return {
        "message": "Link valid only for 5 min .",
        "Download url": download_urls
    }

async def delete_pdf_path(pdf_file_path ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    if os.path.exists(pdf_file_path):
            os.remove(pdf_file_path)

@app.get("/download/pdf/{image_to_pdf_file}/{file_name}")
async def download_pdf(image_to_pdf_file: str , file_name: str):

    pdf_path = os.path.join(image_to_pdf_file ,file_name )
    # pdf_path = f"{image_to_pdf_file}/{file_name}"
    if not os.path.exists(pdf_path):

        raise HTTPException(status_code=404, detail="Link Expired")
    
    return FileResponse(pdf_path, headers={"Content-Disposition": f"attachment; filename={pdf_path}"})





# ======================= 1- Image to pdf Directories End  =======================  


# ======================= 2 - Docx to pdf Directories Start  =======================  

@app.post("/convert/doc/to/pdf/")
async def convert_to_pdf(files: list[UploadFile] , background_tasks: BackgroundTasks = BackgroundTasks()):
    # Create the PDF output directory if it doesn't exist
    try:

        download_urls = []
        docx_file_paths = []
        pdf_file_paths = []
        file_name = []

        for file in files:
            file_name = re.sub(r'\s', '_', file.filename)

            # Save the uploaded DOCX file to a temporary location
            docx_file_path = os.path.join( docx_to_pdf_original ,file_name )
            # docx_file_path = f"{pdf_output_dir}/{file_name}"

            with open(docx_file_path, "wb") as f:
                f.write(file.file.read())

            print('docx_file_path :',docx_file_path)

            # Convert the DOCX to PDF using the docx2pdf library
            pdf_file_path = f"{docx_file_path.replace('.docx', '.pdf')}"
            
            print('pdf_file_path :',pdf_file_path)

            # pdf_file_path = f"{pdf_output_dir}/{file_name.replace('.docx', '.pdf')}"
            convert(docx_file_path, pdf_file_path)

            # Construct the download URL for each processed file

            # Schedule a task to delete the image after 1 minute
            docx_file_paths.append(docx_file_path)
            pdf_file_paths.append(pdf_file_path)
            print('pdf_file_paths :',pdf_file_paths)

            pdf_file_name = pdf_file_path.split('\\')[-1]
            # print('pdf_file_path :',pdf_file_path)

            download_url = f"{server_url}/download/converted/pdf/{pdf_file_name}"
            download_urls.append(download_url)


        background_tasks.add_task(delete_doc_and_pdf, docx_file_paths , pdf_file_paths)

        return {"message": "Files converted to PDF", "download_urls": download_urls}
    
    
    except Exception as e:
        background_tasks.add_task(delete_doc_and_pdf, docx_file_paths , pdf_file_paths)

        raise HTTPException(status_code=500, detail=str(e))
    

    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_filename)

async def delete_doc_and_pdf(docx_file_paths , pdf_file_paths ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)
    
    for path in docx_file_paths:
        if os.path.exists(path):
                print("file removed  :",path)
                os.remove(path)

    for path in pdf_file_paths:
        if os.path.exists(path):
                print("file removed  :",path)
                os.remove(path)


@app.get("/download/converted/pdf/{pdf_file_name}")
async def download_image(pdf_file_name: str ):
    
    pdf_path = os.path.join('docx_to_pdf_original', pdf_file_name)
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Link Expired")

    response = FileResponse(pdf_path, headers={"Content-Disposition": f"attachment; filename={pdf_file_name}"})

    return response

# ======================= 2 - Docx to pdf Directories End  =======================  


# ======================= 3 - Pdf to Docx Directories Start  =======================  


@app.post("/convert/pdf/to/docx/")
async def convert_to_docx(files: list[UploadFile], background_tasks: BackgroundTasks):
    try:
        download_urls = []
        pdf_file_paths = []
        docx_file_paths = []

        for file in files:
            pdf_file_name = re.sub(r'\s', '_', file.filename)

            # Save the uploaded PDF file to a temporary location
            pdf_file_path = os.path.join(pdf_to_docx_original, pdf_file_name)

            with open(pdf_file_path, "wb") as f:
                f.write(file.file.read())

            print('pdf_file_path:', pdf_file_path)

            # Convert the PDF to DOCX
            docx_file_name = pdf_file_name.replace('.pdf', '.docx')
            docx_file_path = os.path.join(pdf_to_docx_original, docx_file_name)

            # Use pdf2docx to perform the conversion
            cv = Converter(pdf_file_path)
            cv.convert(docx_file_path, start=0, end=None)
            cv.close()

            # Construct the download URL for each processed file
            pdf_file_paths.append(pdf_file_path)
            docx_file_paths.append(docx_file_path)

            download_url = f"{server_url}/download/converted/docx/{docx_file_name}"
            download_urls.append(download_url)

        background_tasks.add_task(delete_pdf_and_docx, pdf_file_paths, docx_file_paths)

        return {"message": "Files converted to DOCX", "download_urls": download_urls}

    except Exception as e:
        background_tasks.add_task(delete_pdf_and_docx, pdf_file_paths, docx_file_paths)
        raise HTTPException(status_code=500, detail=str(e))

async def delete_pdf_and_docx(pdf_file_paths, docx_file_paths):
    await asyncio.sleep(300)  # Wait for 300 seconds (5 minutes)

    for path in pdf_file_paths:
        if os.path.exists(path):
            print("file removed:", path)
            os.remove(path)

    for path in docx_file_paths:
        if os.path.exists(path):
            print("file removed:", path)
            os.remove(path)

@app.get("/download/converted/docx/{docx_file_name}")
async def download_docx(docx_file_name: str):
    docx_path = os.path.join(pdf_to_docx_original, docx_file_name)

    if not os.path.exists(docx_path):
        raise HTTPException(status_code=404, detail="Link Expired")

    response = FileResponse(docx_path, headers={"Content-Disposition": f"attachment; filename={docx_file_name}"})

    return response


# ======================= 3 - Pdf to Docx Directories End  =======================  


# ======================= 4 - Remove Image Directory Start =======================

def remove_background(input_image_path, output_image_path, th=None):
    # print('th:', th)
    with open(input_image_path, 'rb') as f:
        input_data = f.read()

        if th:
            subject = remove(input_data, alpha_matting=True, alpha_matting_foreground_threshold=th)
        else:
            subject = remove(input_data)

    with open(output_image_path, 'wb') as f:
        f.write(subject)


@app.post("/api/remove-background/")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks(), th: int = Form(None)):
    try:
        # os.makedirs('original', exist_ok=True)
        # os.makedirs('static', exist_ok=True)
        
        img_name = file.filename

        img_path = os.path.join(image_remove_bg_original, img_name)
        output_path = os.path.join(image_remove_bg_final, img_name)

        with open(img_path, "wb") as image_file:
            shutil.copyfileobj(file.file, image_file)


        remove_background(input_image_path=img_path, output_image_path=output_path, th=th)

        # Construct the HTTP URL for the masked image
        masked_image_url = f"{server_url}/{image_remove_bg_final}/{img_name}"

        # Construct the download URL for the processed image
        download_url = f"{server_url}/download-image/{img_name}"

        # Schedule a task to delete the image after 1 minute
        background_tasks.add_task(delete_image_bg, img_path, output_path)

        return {"download_url": download_url , "message": "Download Link valid for 5 minuts ."}

    except Exception as e:
        return {"error": str(e)}
    

async def delete_image_bg(img_path, original_img_path ,**kwargs ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    # print("kwargs['path'] :",kwargs['path'])

    if os.path.exists(img_path):
        os.remove(img_path)

    if os.path.exists(original_img_path):
        os.remove(original_img_path)

    # if os.path.exists(kwargs['path']):
    #     os.remove(kwargs['path'])

@app.get("/download-image/{img_name}")
async def download_image(img_name: str):
    img_path = os.path.join(image_remove_bg_final, img_name)
    
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Link Expired")

    response = FileResponse(img_path, headers={"Content-Disposition": f"attachment; filename={img_name}"})

    return response


# ======================= 4 - Remove Image Directory end =======================

# ======================= 5 - Replace background  start =======================

@app.post("/api/replace-background/")
async def upload_bgimage(image: UploadFile = File(...), background: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks(), th: int = Form(None)):
    try:
        
        # os.makedirs('original', exist_ok=True)
        # os.makedirs('RemoveBackground', exist_ok=True)
        # os.makedirs('ReplaceBackground', exist_ok=True)

        img_name = image.filename

        img_path = os.path.join(RemoveBackground_replace_bg , img_name)
        output_path = os.path.join( RemoveBackground_replace_bg , img_name)
        # replace_path = os.path.join('ReplaceBackground' , img_name)

        with open(img_path, "wb") as image_file:
            shutil.copyfileobj(image.file, image_file)


        # remove_background(input_image_path=img_path, output_image_path=output_path, th=th)
        
        with open(img_path, 'rb') as f:
            input_data = f.read()
            subject = remove(input_data)

        with open(output_path, 'wb') as f:
            f.write(subject)

        img = Image.open(image.file)
        bg_img = Image.open(background.file)
        bg_name = background.filename
        # print(img , type(img))

        bg_img = bg_img.resize((img.width , img.height))


        fg_img = Image.open(output_path)

        bg_img.paste(fg_img , (0,0) , fg_img)
        bg_img.save(f"{ReplaceBackground}/{img_name}" , format='jpeg')

        # bg_img.show()

        # Construct the download URL for the processed image
        download_url = f"{server_url}/download/replace/bg-image/{img_name}"

        # Schedule a task to delete the image after 1 minute
        background_tasks.add_task(delete_replace_bg_image_, img_path, output_path  )

        return {"download_url": download_url , "message": "Download Link valid for 5 minuts ."}

    except Exception as e:
        return {"error": str(e)}


async def delete_replace_bg_image_(img_path, original_img_path ,**kwargs ):
    await asyncio.sleep(300)  # Wait for 500 seconds (5 minute)

    # print("kwargs['path'] :",kwargs['path'])

    if os.path.exists(img_path):
        os.remove(img_path)

    if os.path.exists(original_img_path):
        os.remove(original_img_path)

    # if os.path.exists(kwargs['path']):
    #     os.remove(kwargs['path'])


@app.get("/download/replace/bg-image/{img_name}")
async def download_replacebg_image(img_name: str):
    img_path = os.path.join(ReplaceBackground, img_name)
    
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Link Expired")

    response = FileResponse(img_path, headers={"Content-Disposition": f"attachment; filename={img_name}"})

    return response


# ======================= 5 - Replace background  end =======================


# ======================= 6 - Extract Text from Image Start =======================


reader = easyocr.Reader(['en'])
@app.post("/api/extract/text/from-image/")
async def upload_image_and_extract_text(image: UploadFile):
    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Use easyocr to extract text from the image
        results = reader.readtext(image_data)

        # Extract and concatenate the text from the results
        extracted_text = " ".join(result[1] for result in results)

        return {"text": extracted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================= 6 - Extract Text from Image End =======================



# ======================= 7 - Text to audio Start =======================


def modify_audio(audio, option):
    if option == 1:  # drunk
        audio = audio.speedup(playback_speed=0.7)
    elif option == 2:  # Reverse
        audio = audio.reverse()
    # Add more options and modifications here...

    return audio

@app.post("/convert_text_to_audio/{option}")
async def convert_text_to_audio(text: str, option: int):
    if option < 1 or option > 18:
        raise HTTPException(status_code=400, detail="Option out of range. Choose between 1-18")

    # Convert text to speech using gTTS
    tts = gTTS(text)
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)

    # Convert gTTS audio to pydub AudioSegment
    audio = AudioSegment.from_file(audio_io, format="mp3")

    # Modify the audio based on the selected option
    modified_audio = modify_audio(audio, option)

    # Export the modified audio to a byte object
    modified_audio_io = io.BytesIO()
    modified_audio.export(modified_audio_io, format="mp3")
    modified_audio_io.seek(0)

    # Return the modified audio
    return {"audio_data": modified_audio_io}



# ======================= 7 - Text to audio End =======================




# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)

