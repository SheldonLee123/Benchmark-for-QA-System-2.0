import csv
import json
import pandas as pd
from tqdm import trange
from multiprocessing import Pool

def checkID(uid, ans_json):
    with open(ans_json, "r", encoding="utf-8") as json_file:
        ls = json.load(json_file)
        flag = False
        for item in ls:
            ans_id = item['uid']
            if ans_id == uid:
                flag = True
                return flag
        return flag

def getTemporalData(csvfile, ans_json):
    df = pd.read_csv(csvfile)
    uid = df.uid
    list = []
    for i in trange(len(uid)):
        flag = checkID(uid[i], ans_json)
        if not flag:
            list.append(uid[i])
    return list

def mycallback(list):
    output = "./dataset/lcquad2/Uidnotin.csv"
    with open(output, "a+", encoding="utf-8", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in list:
            csv_writer.writerow([i])

if __name__ == '__main__':
    csvfile = "./dataset/lcquad2/output/id-temporal.csv"
    ans_json = "./dataset/lcquad2/TempQuestionV2_QidAnswer_LcQuadV2_total_ans.json"

    output = "./dataset/lcquad2/Uidnotin.csv"
    with open(output, "w", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['uid'])
        csv_file.close()

    tempfile = []
    for i in range(8):
        tempfile.append("./dataset/lcquad2/output/id-temporal" + str(i) + ".csv")

    pool = Pool()
    for i in range(8):
        pool.apply_async(getTemporalData, (tempfile[i],ans_json, ), callback=mycallback)
    pool.close()
    pool.join()

    # getTemporalData(csvfile, output, ans_json)