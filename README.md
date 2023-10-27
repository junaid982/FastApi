# FastApi
this FastApi contains 6 endpoints 


This is a fast api contains 7 endpoints

1 - root ( '/' )
to test the server is working or not 

its return Json {'404':'Server not found'}


2 - convert imgaes to pdf ('http://127.0.0.1:8000/convert/img/to/pdf/ ') 
its required a POST request

and FormData Body 

key : images   , value : image_files

its return a pdf download link which is valid for 5 min 


3 - convert docx to pdf ('http://127.0.0.1:8000/convert/doc/to/pdf/ ')
its required a POST request

and FormData Body 

key : files   , value : docx file 

its return a pdf download link which is valid for 5 min 
this endpoint is capable to convert single or multiple docx to pdf 


4 - convert pdf to docx ('http://127.0.0.1:8000/convert/pdf/to/docx/ ')
its required a POST request

and FormData Body 

key : files   , value : pdf file 

its return a docx download link which is valid for 5 min 
this endpoint is capable to convert single or multiple pdf to docx


5 - Remove image background ('http://127.0.0.1:8000/api/remove-background/')

its required a POST request

and FormData Body 

key : file   , value : image

This endpoint remove image background and return a download link which is valid for 5 min




6 - Replace image background ('http://127.0.0.1:8000/api/remove-background/')

its required a POST request

and FormData Body 

key : image   , value : image_file
key : background , value L bg_image

This endpoint replace image background to bg_image and return a download link which is valid for 5 min




7 - Extract Text from image ('http://127.0.0.1:8000/api/extract/text/from-image/')
its required a POST request


and FormData Body 

key : image , value : image_file_with_text


This endpoint will extract the text and return a extracted text in a string 



































