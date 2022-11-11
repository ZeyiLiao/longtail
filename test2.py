import sys

def main_test2(l):
    print('at main_test2())',__name__)
    print(l[1])

if __name__ == '__main__':
    print('at __main__',__name__)
    main_test2(sys.argv)