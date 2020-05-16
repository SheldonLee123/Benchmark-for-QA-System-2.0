# coding: utf-8
import csv
from tqdm import trange

def TxtToCsv(txtfile, csvfile):
	with open(txtfile, "r", encoding='UTF-8') as f:
		while True:
			lines = f.readline()
			if not lines:
				break
			lines = lines.strip()
			print(lines)
			print(lines.split("|||")[1])

		# with open(csvfile, "w", encoding='UTF-8', newline='') as csv_file:
		# 	csv_writer = csv.writer(csv_file)
		# 	csv_writer.writerow(["id", "question"])
		# 	while True:
		# 		lines = f.readline()
		# 		if not lines:
		# 			break
		# 		print(lines)
		# 		t_id, question = lines.split("|||")
		# 		csv_writer.writerow([t_id, question])

def Combine_CSV():
	csv1 = "./dataset/lcquad2/output/id-Explicit.csv"
	csv2 = "./dataset/lcquad2/output/id-Implicit.csv"
	csv3 = "./dataset/lcquad2/output/id-Ordinal.csv"
	csv4 = "./dataset/lcquad2/output/id-Temp.Ans.csv"
	new_csv = "./dataset/lcquad2/output/id-ClassAll.csv"
	with open(new_csv, "w", encoding='UTF-8', newline='') as new_csv_file:
		pass
	with open(new_csv, "a", encoding='UTF-8', newline='') as new_csv_file:
		csv_writer = csv.writer(new_csv_file)
		csv_writer.writerow(["id", "question", "tempoInfo"])
		with open(csv1, "r", encoding='UTF-8') as csv1_file:
			reader = list(csv.reader(csv1_file))
			for i in trange(len(reader)):
				id = reader[i][0]
				question = reader[i][1]
				tempoInfo = "Explicit"
				csv_writer.writerow([id, question, tempoInfo])
		with open(csv2, "r", encoding='UTF-8') as csv2_file:
			reader = csv.reader(csv2_file)
			for row in reader:
				id = row[0]
				question = row[1]
				tempoInfo = "Implicit"
				csv_writer.writerow([id, question, tempoInfo])
		with open(csv3, "r", encoding='UTF-8') as csv3_file:
			reader = csv.reader(csv3_file)
			for row in reader:
				id = row[0]
				question = row[1]
				tempoInfo = "Ordinal"
				csv_writer.writerow([id, question, tempoInfo])
		with open(csv4, "r", encoding='UTF-8') as csv4_file:
			reader = csv.reader(csv4_file)
			for row in reader:
				id = row[0]
				question = row[1]
				tempoInfo = "Temp.Ans"
				csv_writer.writerow([id, question, tempoInfo])

if __name__ == '__main__':
    TxtToCsv("./dataset/lcquad2/output/id-temporal.txt", "./dataset/lcquad2/output/id-temporal.csv")
    # TxtToCsv("./dataset/lcquad2/output/id-Explicit.txt", "./dataset/lcquad2/output/id-Explicit.csv")
    # TxtToCsv("./dataset/lcquad2/output/id-Implicit.txt", "./dataset/lcquad2/output/id-Implicit.csv")
    # TxtToCsv("./dataset/lcquad2/output/id-Ordinal.txt", "./dataset/lcquad2/output/id-Ordinal.csv")
    # TxtToCsv("./dataset/lcquad2/output/id-Temp.Ans.txt", "./dataset/lcquad2/output/id-Temp.Ans.csv")
	# Combine_CSV()