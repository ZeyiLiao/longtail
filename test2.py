from sympy.logic.boolalg import to_cnf
from sympy import *


test_expr = '(c & d) | (c & e) | (c & f) | (d & e) | (d & f) | (e & f)'
expr = sympify(test_expr)
print(to_cnf(expr,True))
# D = False
# E = False
# print(to_cnf((A & B) | (A & C) | (A & D) | (A & E) | (B & C) | (B & D) | (B & E) | (C & D) | (C & E) | (D & E), True))
# print(to_cnf((A & B) | (A & C) | (B & C), True))