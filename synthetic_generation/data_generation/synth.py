import os
from random_tex_bot import TexBot

tex_dir = '../data/tex_files'
build_dir = '../data/build_files'

texbot = TexBot()

for i in range(50):
    tex = texbot.gen_completion(i)
    print(tex)
    print()

    file_path = os.path.join(tex_dir, f'{i}.tex')
    with open(file_path, 'w+') as f:        
        f.write(tex)