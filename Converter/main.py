
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import tempfile
import weasyprint

app = FastAPI()


server_url = 'http://127.0.0.1:8000'

# Directory where uploaded .docx files will be stored temporarily
docx_to_pdf_original = "docx_to_pdf_original"
docx_to_pdf_output = "docx_to_pdf_output"

if not os.path.exists(docx_to_pdf_original):
    os.makedirs(docx_to_pdf_original)

if not os.path.exists(docx_to_pdf_output):
    os.makedirs(docx_to_pdf_output)


@app.post("/convert/doc/to/pdf/")
async def convert_doc_to_pdf(file: UploadFile):
    try:
        # Save the uploaded .docx file
        with open(os.path.join(docx_to_pdf_original, file.filename), "wb") as f:
            f.write(file.file.read())

        pdf_filename = os.path.splitext(file.filename)[0] + ".pdf"
        pdf_path = os.path.join(docx_to_pdf_output, pdf_filename)

        # Convert .docx to .pdf using WeasyPrint
        weasyprint.HTML(string=f"<html><body><p>{file.filename}</p></body></html>").write_pdf(pdf_path)

        download_link = f"{server_url}/getpdf/{pdf_filename}"

        return {"pdf_file_path": pdf_path, "download_link": download_link}
    except Exception as e:
        print("Error during conversion:", str(e))
        raise HTTPException(status_code=500, detail="Conversion failed")

@app.get("/getpdf/{pdf_filename}")
async def get_pdf(pdf_filename: str):
    pdf_path = os.path.join(docx_to_pdf_output, pdf_filename)
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, headers={"Content-Disposition": f"attachment; filename={pdf_filename}"})
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




