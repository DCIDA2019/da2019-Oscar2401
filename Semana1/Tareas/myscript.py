#-------------------------------------
# file: myscript.py

def my_square(x):
    """square a number"""
    return x ** 2

for N in range(1, 4):
    print(N, "squared is", my_square(N))