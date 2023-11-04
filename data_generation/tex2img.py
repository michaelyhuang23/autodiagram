import os
from pdf2image.pdf2image import convert_from_path

tex_dir = '../data/tex_files'
build_dir = '../data/build_files'
img_dir = '../data/img_files'

for file in os.listdir(tex_dir):
    if not file.endswith('.tex'): continue
    file_path = os.path.join(tex_dir, file)
    os.system(f'pdflatex -interaction=nonstopmode -output-directory {build_dir} {file_path}')

for file in os.listdir(build_dir):
    if not file.endswith('.pdf'): continue
    file_path = os.path.join(build_dir, file)
    name = file.split('.')[0]
    images = convert_from_path(file_path)
    images[0].save(os.path.join(img_dir, f'{name}.jpg'), 'JPEG')
