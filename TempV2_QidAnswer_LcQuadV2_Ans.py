import re
import time
import requests
import codecs
import json
from suds.client import Client
import pickle
from socket import timeout
import io

"""
"uid": 8669,
"sparql": "SELECT ?obj WHERE { wd:Q42168 p:P1082 ?s . ?s ps:P1082 ?obj . ?s pq:P585 ?x filter(contains(YEAR(?x),'2013')) }",
"NNQT_ques": "What is  population  of  Clermont-Ferrand  that is  point in time  is  2013-1-1  ?",
"NNQT_ques_tempoInfo": "during|||Explicit;",
"ques": "What was the population of Clermont-Ferrand on 1-1-2013?",
"ques_tempoInfo": "on|||Explicit;",
"para_ques": "How many people lived in Clermont-Ferrand on January 1st, 2013?",
"para_ques_tempoInfo": "on|||Explicit;"

"Id": 2603,
"Question": "who is the monarch of united kingdom before elizabeth ii?",
"Temporal signal": [
			"BEFORE"
		],
"Type": [
			"Implicit"
		],
"Answer": [
			{
				"AnswerType": "Entity",
				"WikidataQid": "Q280856",
				"WikidataLabel": "George VI",
				"WikipediaURL": "https://en.wikipedia.org/wiki/george_vi"
			}
		],
"Data source": "ComQA",
"Question creation date": "NAACL2019(2019-06-03)"
"""
ans_dic = {"Id":"","Question":"","Temporal signal":[],"Type":[],"Answer":"","Data source":"","Question creation date":""}
in_dic = {"uid":"","sparql":"","NNQT_ques":"","NNQT_ques_tempoInfo":"","ques":"","ques_tempoInfo":"","para_ques":"","para_ques_tempoInfo":""}

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
		# print(query_result)
		if 'results' in query_result:
			result = query_result['results']['bindings']
		elif 'boolean' in query_result:
			result = query_result['boolean']
		# time.sleep(5)
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

def find_answer(query):
	answers = []
	results = get_query_request(query)
	"""
	[{'obj': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q43044'}, 'value1': {'datatype': 'http://www.w3.org/2001/XMLSchema#dateTime', 'type': 'literal', 'value': '2005-09-24T00:00:00Z'}}, {'obj': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q37628'}, 'value1': {'datatype': 'http://www.w3.org/2001/XMLSchema#dateTime', 'type': 'literal', 'value': '2015-01-01T00:00:00Z'}}]

	"""
	if type(results) == bool:
		answers.append(results)
	else:
		if len(results) > 0:
			for res in results:
				answer_set = []
				for key in res.keys():
					dic = {}
					dic['value'] = res[key]['value']
					dic['type'] = res[key]['type']
					if dic['type'] == 'uri' and dic['value'].startswith('http://www.wikidata.org/entity/'):
						wiki_label = find_label_wikipedialink(dic['value'])
						if '|||' in wiki_label:
							dic['wikilink'] = wiki_label.split('|||')[0]
							dic['label'] = wiki_label.split('|||')[1]
							print(dic['wikilink'], dic['label'])
					answer_set.append(dic)
				answers.append(answer_set)
	return answers


#in_dic = {"uid":"","sparql":"","NNQT_ques":"","NNQT_ques_tempoInfo":"","ques":"","ques_tempoInfo":"","para_ques":"","para_ques_tempoInfo":""}

def get_label_wikilink(tempLcfile,outputfile):
	with io.open(tempLcfile, encoding='utf-8') as json_data:
		items = json.load(json_data)
		# print(len(items))     # 4996
		json_data.close()

	resultjson = codecs.open(outputfile, "w", "utf-8")
	resultjson.close()
	times = 0
	with io.open(outputfile, "w", encoding="utf-8") as resultjson:
		for item in items:
			query = item["sparql_wikidata"].strip()
			res = find_answer(query)
			item["answer"] = res
			resultjson.write(json.dumps(item, indent=4, ensure_ascii=False))
			times = times + 1
			print(times)
			# if times == 20:
			# 	break
		resultjson.close()

# tempLcfile = r'G:\QA\QA_corpus\other-corpus\otherCorpus\LCQUAD-2' + "\\TempQuestionV2_QidAnswer_LcQuadV2_test.json"
# outputfile = r'G:\QA\QA_corpus\other-corpus\otherCorpus\LCQUAD-2' + "\\TempQuestionV2_QidAnswer_LcQuadV2_test_ans.json"

linux_tempLcfile = "./dataset/lcquad2/lcquad_2_0.json"
# linux_tempLcfile = "/home/sheldon/Documents/Keras/FinalProject/train.json"
linux_outputfile = "./dataset/lcquad2/lcquad_2_0_Ans.json"

get_label_wikilink(linux_tempLcfile,linux_outputfile)

# get_label_wikilink(tempLcfile,outputfile)

# print (find_answer("SELECT ?answer WHERE { wd:Q406643 wdt:P2283 ?X . ?X wdt:P1535 ?answer}"))
# print (find_answer("ASK WHERE { wd:Q178903 wdt:P106 wd:Q40348 }"))
