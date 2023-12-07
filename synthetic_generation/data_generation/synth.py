import os
from random_tex_bot import TexBot

tex_dir = '../data3/tex_files'
build_dir = '../data3/build_files'

texbot = TexBot()

for i in range(10):
    tex = texbot.gen_completion(i)
    print(tex)
    print()

    file_path = os.path.join(tex_dir, f'{i}.tex')
    with open(file_path, 'w+') as f:        
        f.write(tex)
