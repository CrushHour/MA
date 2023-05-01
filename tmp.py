# a function that gives back the midnight formular solution for a given a,b,c
# a,b,c are integers
# a,b,c are the coefficients of the equation ax^2 + bx + c = 0
# x1,x2 are the solutions of the equation
# x1,x2 are complex numbers
def midnight(a,b,c):
    x1 = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    x2 = (-b - (b**2 - 4*a*c)**(1/2))/(2*a)
    return x1,x2