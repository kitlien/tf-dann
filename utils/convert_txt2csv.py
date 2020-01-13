import csv
file_name = '../headshot_landmarks/train_list_2w.csv'
txt_filename = '../headshot_landmarks/train_list_2w.txt'
with open(file_name, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, dialect = 'excel')
    with open(txt_filename,'rb') as file_txt:
        for line in file_txt:
            line_data = line.strip().split(' ')
            spamwriter.writerow(line_data)