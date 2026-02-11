import pandas as pd

class TabFact():
    def __init__(self, data_path, data_testset):
        self.table_info = self._get_data_(data_path, data_testset)
        self.table_path = data_path
    
    def _get_data_(self, data_path, data_testset):
        tabqa = pd.read_csv(data_path + data_testset,on_bad_lines='skip')
        tabqa = tabqa.to_dict(orient='records')
        return tabqa
    
    def get_table_info(self, entry):
        table_file = self.table_path + entry['table_id']
        # .replace('.csv','.tsv')
        
        table = pd.read_csv(table_file, sep='#', encoding='utf-8', engine='python')
        return entry['index'], table, '', entry['statement'], entry['label']
