from datetime import datetime
from uuid import uuid4
import csv



class LogManager():
    def __init__(self, active:bool = True):         
        self.unique_id = uuid4()
        self.log_file = f'logs/{self.unique_id}.log'
        self.csv_file = f'logs/{self.unique_id}.csv'
        self.active = active
        
        # Create log file
        with open(self.log_file, 'w') as file:
            file.write(f'Log file created at {datetime.now()}')
            file.write(f'\n\nLog id: {self.unique_id}\n\n')
            print(f'Log file created at {datetime.now()}')
            print(f'Log id: {self.unique_id}\n\n')   
        # Define csv headers
        headers = ['generation', 'individual', 'fitness', 'compression_ratio','integrity_check']
        with open(self.csv_file, mode='w') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
               

    def add_log(self, log:str):            
        if self.active:
            with open(self.log_file, 'a') as file:                    
                file.write(f'\n Time: {datetime.now()}, Log:{log}')

    def add_csv_row(self, data:dict):
        with open(self.csv_file, mode='a') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writerow(data)


    
    
    

    

    
        