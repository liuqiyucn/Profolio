from sympy import symbols, init_printing

# Initialize pretty printing
init_printing()

a, b = symbols('a b')
expr = a / b

# Print the expression
print(expr)

