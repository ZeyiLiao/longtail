import sys
from test2 import main_test2

def main_test1(l):
    print('at main_test1()',__name__)
    print(l[1])

if __name__ == '__main__':
    print('at __main__',__name__)
    main_test2(sys.argv)