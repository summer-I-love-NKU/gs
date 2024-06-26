
import csv

def run_res(dataset_name='ka',useNE=0,uselazy=0):
    with open(f'RES_{dataset_name}_useNE_{useNE}_uselazy_{uselazy}.csv', mode='r') as file:
        reader = csv.reader(file)
        # for row in reader:
        #     print(row)
        data = list(reader)
    data=sorted(data,key=lambda x:float(x[0]))
    # print(data)
    transposed_data = [list(row) for row in zip(*data)]
    # print(transposed_data)
    with open(f'E:\desktop\RES_{dataset_name}_new.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(transposed_data)


run_res(dataset_name='ka',useNE=1,uselazy=0)