import os

from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

tex_dir = '../data/tex_files'
build_dir = '../data/build_files'
img_dir = '../data/img_files'

for file in os.listdir(build_dir):
    if not file.endswith('.pdf'): continue
    file_path = os.path.join(build_dir, file)
    name = file.split('.')[0]
    images = convert_from_path(file_path)
    images[0].save(os.path.join(img_dir, f'{name}.jpg'), 'JPEG')

