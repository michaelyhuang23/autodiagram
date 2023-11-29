import os
from pdf2image.pdf2image import convert_from_path

tex_dir = '../data/tex_files'
build_dir = '../data/build_files'
img_dir = '../data/img_files'

for file in os.listdir(tex_dir):
    if not file.endswith('.tex'): continue
    file_path = os.path.join(tex_dir, file)
    os.system(f'pdflatex -interaction=nonstopmode -output-directory {build_dir} {file_path}')