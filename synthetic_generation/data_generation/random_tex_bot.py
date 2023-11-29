import random

class TexBot:
    def __init__(self) -> None:
        self.N = 2
        pass

    # Draw Rectangles
    def draw_rectangle(self):
        N = self.N
        x0 = N*random.random()
        x1 = N*random.random()
        y0 = N + N * random.random()
        y1 = N + N * random.random()
        return f"\draw ({x0},{y0}) rectangle ({x1},{y1}); \n"
    
    # Draw Triangles
    def draw_triangle(self):
        N = self.N
        x0 = N*random.random()
        x1 = N*random.random()
        x2 = N*random.random()
        y0 = N * random.random()
        y1 = N * random.random()
        y2 = N * random.random()
        return f"\draw ({x0},{y0}) -- ({x1},{y1}) -- ({x2},{y2}) -- cycle; \n"

    # Draw Arrows
    def draw_arrow(self):
        N = self.N
        x0 = N*random.random()
        x1 = N*random.random()
        y0 = N + N * random.random()
        y1 = N + N * random.random()
        return f"\draw[->] ({x0},{y0}) -- ({x1},{y1}); \n"

    # Draw Circles
    def draw_circle(self):
        N = self.N
        x0 = N+N/4*random.random()
        y0 = N+ N/4 * random.random() 
        r = N * random.random() 

        return f"\draw ({x0},{y0}) circle ({r}); \n"

    def gen_main_body(self, seed):
        random.seed(seed)

        num_objects = random.randint(3,8)
        objects = ""
        for _ in range(num_objects):
            x = random.randint(1,4)

            # I have python 3.8.5 and I can't use switch statements so that's why
            print(x)
            if(x==1):
                new_obj = self.draw_rectangle()
                print("rectangle")
            if(x==2):
                new_obj = self.draw_triangle()
                print("triangle")
            if(x==3):
                new_obj = self.draw_arrow()
            if(x==4):
                new_obj = self.draw_circle()
            
            objects += new_obj
        return objects


    def gen_completion(self, seed):
        head = r"""\documentclass[tikz,border=3mm]{standalone}
\begin{document}
\begin{tikzpicture}
""" 
        body = self.gen_main_body(seed)

        tail = r"""\end{tikzpicture}
\end{document}"""

        return head + body + tail