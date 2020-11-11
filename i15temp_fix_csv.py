import csv
import os


def csv_printer_from_localfile(filename, directory='./'):
    mylist = []
    with open(os.path.join(directory, filename), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        lastrow = None
        for row in reader:
            # print(len(row))
            # print(row[0], row[1], row[2])
            # for idx, val in enumerate(row):
            #     print(idx, val)
        # ["admin", "admin123", "10.248.10.100:554", 1, 'dahua', "华新数创5楼室内0", "80002302|00038910", 'YHat, BHat, NoHat']
            ilist = [ row[1], row[2], row[3]+":554", 1, "hik", row[6], "80002302|00038910|" + row[7], "RHat|YHat|BHat|NoHat"]
            print(ilist)
            print(',')
            mylist.append(ilist)
    print('\n\n')
    print(mylist)

if __name__ == "__main__":
    csv_printer_from_localfile('i13rtsp_list.csv', '/Users/abel/Desktop/' )