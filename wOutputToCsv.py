import numpy as np
import csv


class OW:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def write_numpy_array_to_csv(self, array):
        np.savetxt(self.csv_path, array, delimiter=',')

    def append_numpy_array_to_csv(self, array):
        with open(self.csv_path, 'a') as csvfile:
            np.savetxt(csvfile, array, delimiter=',')

    def read_numpy_array_from_csv(self):
        return np.genfromtxt(self.csv_path, delimiter=',')
    
if __name__ == "__main__":
    # Example usage
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    csv_path = 'OutputCsv/data.csv'
    my_class = OW(csv_path)

    my_class.write_numpy_array_to_csv(array)

    new_array = np.array([[10, 11, 12]])
    my_class.append_numpy_array_to_csv(new_array)

    data = my_class.read_numpy_array_from_csv()
    print(data)