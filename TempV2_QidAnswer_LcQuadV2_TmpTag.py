import re
import time
import requests
import codecs
import json
from suds.client import Client
import pickle
from socket import timeout

qid_map_wiki = {}
def get_query_request(sparql_query):
    """
        access wikidata end point.
        :param sparql_query
        :return:
        """
    result = []
    try:
        r = requests.get("https://query.wikidata.org/sparql", params={'format': 'json', 'query': sparql_query, })
        r.encoding = 'utf-8'
        query_result = r.json()
        result = query_result['results']['bindings']
        time.sleep(3)
    except:
        print("wikidata endpoint query exception!")
    return result

def find_label_wikipedialink(qid):
    if qid in qid_map_wiki.keys():
        return qid_map_wiki[qid]

    query = 'SELECT distinct ?qid ?qidLabel ?wikipedia_link where{ '
    query += '?wikipedia_link' + ' schema:about ' + '?qid' + '. '
    query += '?wikipedia_link' + ' schema:isPartOf <https://en.wikipedia.org/>. '
    query += '?wikipedia_link' + ' schema:inLanguage "en" .'
    query += 'FILTER ( ?qid = <' + qid + '> )'
    query += "SERVICE wikibase:label {bd:serviceParam wikibase:language \"en\".} "
    query += 'OPTIONAL { ?qid  wdt:P569  ?DR }'
    query += 'OPTIONAL{ ?qid  wdt:P570  ?RIP }'
    query += 'OPTIONAL{ ?qid  wdt:P18  ?image }}'
    print(query)
    results = get_query_request(query)
    if len(results) > 0:
        wikipedia_link = results[0]['wikipedia_link']['value']
        label = results[0]['qidLabel']['value']
        qid_map_wiki[qid] = wikipedia_link + '|||' + label
    else:
        qid_map_wiki[qid] = "null"
    return qid_map_wiki[qid]


class LcV2Question:
    def __init__(self, Lcoriginalfile):
        self.lcquadcorpus = Lcoriginalfile
        self.lcquestions = []
        self.templcqquestions = []
        self.templcqyesquestions = []


    def get_lccorpus(self):
        with open(self.lcquadcorpus, encoding='utf-8') as json_data:
            corpuslist = json.load(json_data)
            json_data.close()



        for item in corpuslist:
            lcq = {}
            flag = 0
            lcq["uid"] = item["uid"]
            print(lcq["uid"])
            sparql = item["sparql_wikidata"]
            lcq["sparql"] = sparql
            if item["NNQT_question"]:
                lcq["NNQT_ques"] = replace_symbols(item["NNQT_question"])
                NNQT_question_tempoInfo = get_preprocess_result(lcq["NNQT_ques"])
                if NNQT_question_tempoInfo["reasoningWord"] or NNQT_question_tempoInfo["question_type"]:
                    lcq["NNQT_ques_tempoInfo"] = NNQT_question_tempoInfo["reasoningWord"] + "|||" + NNQT_question_tempoInfo[
                    "question_type"]
                    flag = 1
            if item["question"]:
                lcq["ques"] = replace_symbols(item["question"])
                question_tempoInfo = get_preprocess_result(lcq["ques"])
                if question_tempoInfo["reasoningWord"] or question_tempoInfo["question_type"]:
                    lcq["ques_tempoInfo"] = question_tempoInfo["reasoningWord"] + "|||" + question_tempoInfo[
                    "question_type"]
                    flag = 1
            if item["paraphrased_question"]:
                lcq["para_ques"] = replace_symbols(item["paraphrased_question"])
                para_question_tempoInfo = get_preprocess_result(lcq["para_ques"])
                if para_question_tempoInfo["reasoningWord"] or para_question_tempoInfo["question_type"]:
                    lcq["para_ques_tempoInfo"] = para_question_tempoInfo["reasoningWord"] + "|||" + para_question_tempoInfo[
                    "question_type"]
                    flag = 1

            if flag == 1:
                if ((sparql.startswith("S")) or (sparql.startswith("s"))):
                    self.templcqquestions.append(lcq)
                    print(lcq["uid"])
                else:
                    self.templcqyesquestions.append(lcq)

            else:
                self.lcquestions.append(lcq)
                """
                answers = []
                if ((sparql.startswith("S")) or (sparql.startswith("s"))):
                    results = get_query_request(sparql)
                    for res in results:
                        print (res)
                        dic = {}
                        keys = res.keys()
                        for key in keys:
                            dic['value'] = res[key]['value']
                            dic['type'] = res[key]['type']
                            if dic['type'] == 'uri' and dic['value'].startswith('http://www.wikidata.org/entity/Q'):
                                if '|||' in  find_label_wikipedialink(dic['value']):
                                    dic['wikilink']  = find_label_wikipedialink(dic['value']).split('|||')[0]
                                    dic['label']  = find_label_wikipedialink(dic['value']).split('|||')[1]
                                    print (dic['wikilink'],dic['label'])
                            answers.append(dic)
                lcq["answers"] = answers
                self.templcqquestions.append(lcq)
            else:
                self.lcquestions.append(lcq)
                """

        return self.templcqquestions,self.templcqyesquestions,self.lcquestions

def replace_symbols(s):
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('|', ' ')
    s = s.replace('"', ' ')
    s = s.strip()
    return s

def get_preprocess_result(question):
    s1 = replace_symbols(question)
    client = Client("http://sedna.mpi-inf.mpg.de:8991/myWebServiceForQAWordMark?wsdl")
    service = client.service
    client.set_options(timeout=1000)
    tempoInfo = {'reasoningWord':'','question_type':''}
    try:
        response = service.getAnswers(s1)
        preprocess_result = json.loads(response)
        tempoInfo = preprocess_result["TempoInfo"]
    except:
        print ("there is an error when tag temp info!")
    print(tempoInfo)
    return tempoInfo

if __name__ == '__main__':
    #path = '/scratch/GW/pool0/zhen/TeQA_Project/data/preprocess/corpus/'
    winpath = r'G:\QA\QA_corpus\other-corpus\otherCorpus\LCQUAD-2\LC-QuAD2.0-master\dataset'
    lccorpus_train = winpath + "\\train.json"
    lccorpus_test = winpath + "\\test.json"

    outfile1 = r'G:\QA\QA_corpus\other-corpus\otherCorpus\LCQUAD-2' + "\\TempQuestionV2_QidAnswer_LcQuadV2_train.json"
    outfile2 = r'G:\QA\QA_corpus\other-corpus\otherCorpus\LCQUAD-2' + "\\LcQuadV2_tempyes_train.json"
    outfile3 = r'G:\QA\QA_corpus\other-corpus\otherCorpus\LCQUAD-2' + "\\LcQuadV2_nontemp_train.json"

    templcqquestions,templcqyesquestions,lcquestions = LcV2Question(lccorpus_train).get_lccorpus()
    resultjson1 = codecs.open(outfile1, "w", "utf-8")
    resultjson2 = codecs.open(outfile2, "w", "utf-8")
    resultjson3 = codecs.open(outfile3, "w", "utf-8")

    resultjson1.write(json.dumps(templcqquestions, indent=4, ensure_ascii=False))
        #resultjson1.write(',\n')
    resultjson2.write(json.dumps(templcqyesquestions, indent=4, ensure_ascii=False))
        # resultjson2.write(',')
    resultjson3.write(json.dumps(lcquestions, indent=4, ensure_ascii=False))
    resultjson1.close()
    resultjson2.close()
    resultjson3.close()







