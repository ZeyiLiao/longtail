
from itertools import combinations 
from sympy.logic.boolalg import to_cnf
from sympy import *

class Logic_wrapper:
    def __init__(self):
        self.format_dict = {}
        self.map_words_ABC = {}
    

    def map_words_to_ABC(self,words):
        a_id = ord('A')
        all_symbols = []
        for i in range(len(words)):
            symbol = chr(a_id + i)
            all_symbols.append(symbol)
            self.map_words_ABC[symbol] = words[i]
        return all_symbols


    def emulation_combine(self,l,size_combination = 2):
        if self.format_dict.get(len(l),None) is None:
            if len(l) < size_combination:
                size_combination = len(l)
                str = l[0]
                self.format_dict[len(l)] = f'({str})'
                return
            nouns_combation = combinations(l,size_combination)

            str_l = []
            for combination in nouns_combation:
                tmp = f'({combination[0]} & {combination[1]})'
                str_l.append(tmp)
            str = ' | '.join(str_l)
            self.format_dict[len(l)] = str


    def change_to_cnf(self,words):
        expr = sympify(self.format_dict[len(words)])
        cnf_expr = str(to_cnf(expr,True))
        for key in self.map_words_ABC.keys():
            cnf_expr = cnf_expr.replace(key,self.map_words_ABC[key])
        output_l = []
        for cons in cnf_expr.split(' & '):
            cons = cons.strip('(').strip(')')
            tmp = []
            for con in cons.split(' | '):
                tmp.append(con)
            output_l.append(tmp)
        return output_l


    def run(self,words):
        self.map_words_ABC = {}
        all_symbols = self.map_words_to_ABC(words)
        self.emulation_combine(all_symbols)
        return self.change_to_cnf(words)

    