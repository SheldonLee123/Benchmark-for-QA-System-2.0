import csv
import json

def txtTocsv(txtfile, csvfile, label):
	with open(txtfile, "r") as f:
		with open(csvfile, "a") as csv_file:
			csv_writer = csv.writer(csv_file)
			while True:
				lines = f.readline()
				if not lines:
					break
				lines = lines[:-1]
				csv_writer.writerow(['0000', lines, label])
		csv_file.close()
	f.close()

def jsonTocsv(jsonfile, csvfile):
	with open(jsonfile, "r") as json_file:
		with open(csvfile, "a") as csv_file:
			csv_writer = csv.writer(csv_file)
			ls = json.load(json_file)
			print(len(ls))
			for item in ls:
				id = item["uid"]
				NNQT_ques = item["NNQT_ques"]
				csv_writer.writerow([id, NNQT_ques, "positive"])
				if "ques" in item:
					ques = item["ques"]
					csv_writer.writerow([id, ques, "positive"])
				if "para_ques" in item:
					para_ques = item["para_ques"]
					csv_writer.writerow([id, para_ques, "positive"])
		csv_file.close()
	json_file.close()

if __name__ == '__main__':
	with open("./dataset/PosNeg2.0.csv", "w") as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(["id", "question", "label"])
	f.close()
	txtTocsv("./dataset/positive.txt", "./dataset/PosNeg2.0.csv", "positive")
	txtTocsv("./dataset/negative.txt", "./dataset/PosNeg2.0.csv", "negative")
	jsonTocsv("./dataset/lcquad2/TempQuestionV2_QidAnswer_LcQuadV2_same_ans.json", "./dataset/PosNeg2.0.csv")