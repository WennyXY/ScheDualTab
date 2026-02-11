import pandas as pd


class WikiQA():
    def __init__(self, data_path, data_testset):
        self.data = self._get_data_(data_path, data_testset)
        self.table_path = data_path
    
    def _get_data_(self, data_path, data_testset):
        wikitableqa = pd.read_csv(data_path + data_testset, sep="\t", on_bad_lines='skip')
        wikitableqa = wikitableqa.to_dict(orient='records')
        return wikitableqa
    
    def get_table_info(self, entry):
        table_file = entry['context'].replace('.csv','.tsv')
        
        table = pd.read_csv(self.table_path+table_file, sep='\t', encoding='utf-8', engine='python', on_bad_lines='skip')
        return entry['id'], table, '', entry['utterance'], entry['targetValue']
