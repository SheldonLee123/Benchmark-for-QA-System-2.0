import json
import csv
import pandas as pd

# def transcsv(jsonpath, csvpath):
# 	json_file = open(jsonpath, 'r', encoding="utf-8")
# 	csv_file = open(csvpath, 'w', newline="")
# 	ls = json.load(json_file)
# 	data = [list(ls[0].keys())]
# 	for item in ls:
# 		# print(item)
# 		data.append(list(item.values()))
# 	for line in data:
# 		line[1] = '"' + line[1] + '"'
# 		line[2] = str(line[2])
# 		line[2] = '"' + line[2] + '"'
# 		line[3] = str(line[3]).strip('[')
# 		line[3] = line[3].strip(']')
# 		line[3] = line[3].replace("'", '')
# 		# if(line[3][:-1] == "'"):
# 		# 	line[3].strip("'")
# 		print(line)
# 		csv_file.write(",".join('%s' %id for id in line) + "\n")
# 	# 	print(line)
# 	json_file.close()
# 	csv_file.close()

# if __name__ == '__main__':
# 	transcsv("./dataset/TempQuestionV2_QidAnswer_total.json", "./dataset/TempAll.csv")

def transcsv(jsonpath, csvpath):
	json_file = open(jsonpath, 'r', encoding="utf-8")
	csv_file = open(csvpath, 'w', newline="")
	ls = json.load(json_file)
	data = [list(ls[0].keys())]
	for item in ls:
		# print(item)
		data.append(list(item.values()))
	for line in data:
		# print(line)
		line[0] = '"' + line[0] + '"'
		line[2] = '"' + str(line[2]) + '"'
		line[3] = '"' + str(line[3]) + '"'
		line[4] = '"' + str(line[4]) + '"'
		csv_file.write(",".join('%s' %id for id in line) + "\n")
	# 	print(line)
	json_file.close()
	csv_file.close()

def jsontocsv(jsonfile, csvfile):
	with open(jsonfile, "r", encoding="utf-8") as json_file:
		with open(csvfile, "w", encoding="utf-8", newline='') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow(['uid', 'question'])
			ls = json.load(json_file)
			for item in ls:
				uid = item['uid']
				ques = item['question']
				if ques != None and ques != "n/a":
					csv_writer.writerow([uid, ques])

		csv_file.close()
	json_file.close()

if __name__ == '__main__':
	jsonfile = "./dataset/lcquad2/lcquad_2_0.json"
	csvfile = "./dataset/lcquad2/lcquad_2_0_id.csv"
	# transcsv("./dataset/lcquad2/lcquad_2_0.json", "./dataset/lcquad2/lcquad2.csv")
	jsontocsv(jsonfile, csvfile)