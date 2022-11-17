import pickle

class All_Data:
    def __init__(self,all_data_path = '/home/zeyi/longtail/property_centric_process/all_data.pkl'):

        with open(all_data_path,'rb') as f:
            all_data = pickle.load(f)

        self.all_data = all_data
        

    def get_data(self,ids):
        conti_templates = []
        normal_templates = []
        lemmas = []
        inflections = []
        sample_contis = []

        for id in ids:
            conti_template = self.all_data[id]['conti_template']
            normal_template = self.all_data[id]['normal_template']
            lemma = self.all_data[id]['cons_lemma']
            inflection = self.all_data[id]['cons_inflection']
            sample_conti =  self.all_data[id]['sample_cont']

            conti_templates.append(conti_template)
            normal_templates.append(normal_template)
            lemmas.append(lemma)
            inflections.append(inflection)
            sample_contis.append(sample_conti)

        return \
        {"conti_templates":conti_templates,
        "normal_templates":normal_templates,
        "lemmas":lemmas,
        "inflections":inflections,
        "sample_contis":sample_contis}
        