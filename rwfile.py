import csv
import numpy as np
import os.path


class RW:
    
    def __init__(self,csv_path = 'Best/bestlaptime.csv'):
        self.csv_path = csv_path
        if os.path.isfile(self.csv_path) == False:
            print("File doesn't exist.")
            self.write_numpy_array_to_csv(np.array((9993.14)))
            
    
    def write_numpy_array_to_csv(self, array):
        with open(self.csv_path, 'wb') as file:
            np.savetxt(file, np.matrix(array), delimiter=',')
            #writer = csv.writer(csvfile)
            #writer.writerows(array)
    
    def append_numpy_array_to_csv(self, array):
        
        with open(self.csv_path, 'ab') as file:
            np.savetxt(file, np.matrix(array), delimiter=',')
            
    def read_numpy_array_from_csv(self):
            
            float_value = np.loadtxt(self.csv_path , delimiter=',',dtype=float)
            return float_value

if __name__ == "__main__":
    r_w = RW()
    # Usage example
    file_path = 'Best/bestlaptime.csv'  # Replace with the path to your file
    current_float_value = r_w.read_numpy_array_from_csv()
    print("Current float value:", current_float_value)
    
