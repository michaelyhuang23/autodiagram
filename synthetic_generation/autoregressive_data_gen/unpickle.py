import pickle 

with open('line_coords.pickle', 'rb') as f:
    unpickled = []
    while True:
        try:
            unpickled.append(pickle.load(f))
        except EOFError:
            break

print(unpickled)