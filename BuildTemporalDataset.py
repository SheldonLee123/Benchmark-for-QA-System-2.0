import csv
import json
import pandas as pd
from tqdm import trange


def checkInfo(csvfile, id):
    df = pd.read_csv(csvfile)
    for i in range(0, len(df['id'])):
        if id == df['id'][i]:
            return df['tempoInfo'][i]

def buildData(csvfile, jsonfile, new_jsonfile):
    # print(checkInfo(csvfile, "14923"))
    with open(jsonfile, "r", encoding='UTF-8') as json_file:
        ls = json.load(json_file)
        print(len(ls))
        # for i in trange(len(ls)):
        #     ls[i]['tempinfo'] = checkInfo(csvfile, str(ls[i]['uid']))
        # with open(new_jsonfile, "w", encoding='UTF-8', newline='') as new_json_file:
        #     json.dump(ls, new_json_file, indent=4, ensure_ascii=False)

def splitData():
    with open(new_jsonfile, "r", encoding="UTF-8") as new_json_file:
        ls = json.load(new_json_file)
        new_json = []
        count = 1
        for i in ls:
            new_item = {}
            item = {}
            del i['uid']
            if 'ques_tempoInfo' in i:
                del i['ques_tempoInfo']
            if 'NNQT_ques_tempoInfo' in i:
                del i['NNQT_ques_tempoInfo']
            if 'para_ques_tempoInfo' in i:
                del i['para_ques_tempoInfo']
            new_item = i.copy()
            if 'ques' in new_item:
                del new_item['NNQT_ques']
                if 'para_ques' in new_item:
                    del new_item['para_ques']
                item['id'] = count
                count += 1
                item['ques'] = new_item['ques']
                item['sparql'] = new_item['sparql']
                item['answer'] = new_item['answer']
                item['tempinfo'] = new_item['tempinfo']
                new_json.append(item)

            if 'NNQT_ques' in new_item:
                new_item = i.copy()
                if 'ques' in new_item:
                    del new_item['ques']
                if 'para_ques' in new_item:
                    del new_item['para_ques']
                new_item['ques'] = new_item.pop('NNQT_ques')
                item['id'] = count
                count += 1
                item['ques'] = new_item['ques']
                item['sparql'] = new_item['sparql']
                item['answer'] = new_item['answer']
                item['tempinfo'] = new_item['tempinfo']
                new_json.append(item)

            if 'para_ques' in new_item:
                new_item = i.copy()
                if 'ques' in new_item:
                    del new_item['ques']
                del new_item['NNQT_ques']
                new_item['ques'] = new_item.pop('para_ques')
                item['id'] = count
                count += 1
                item['ques'] = new_item['ques']
                item['sparql'] = new_item['sparql']
                item['answer'] = new_item['answer']
                item['tempinfo'] = new_item['tempinfo']
                new_json.append(item)
        for i in new_json:
            print(i)
        # with open(datafile, "w", encoding="UTF-8") as data_file:

if __name__ == '__main__':
    csvfile = "./dataset/lcquad2/output/id-ClassAll.csv"
    jsonfile = "./dataset/lcquad2/TempQuestionV2_QidAnswer_LcQuadV2_same_ans.json"
    new_jsonfile = "./dataset/lcquad2/output/My_lcquad_dataset_with_ans.json"
    datafile = "./dataset/lcquad2/output/My_lcquad_dataset_with_ans2.0.json"

    buildData(csvfile, jsonfile, new_jsonfile)



