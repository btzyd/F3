# coding: utf-8
from evaluation.vqa import VQA
from evaluation.vqaEval import VQAEval
import json
import random
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('result_dir', type=str)
args = parser.parse_args()

# set up file names and paths
# versionType ='' # this should be '' when using VQA v2.0 dataset
# taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
# dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0. 
dataSubType ='val2014'

resultDir = args.result_dir
annFile     = 'evaluation/answer_vqav2_1000.json'
quesFile    = 'evaluation/question_vqav2_1000.json'

imgDir      ='/home/zyd/nfs/VQA_code/VQA-V2/val2014'
resultType  ='rea'
fileTypes   = ['result', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

for prefix in ["clean", "adv", "purify"]:
	[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/%s_%s.json'%(resultDir, prefix, fileType) for fileType in fileTypes]  

	# create vqa object and vqaRes object
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)

	# create vqaEval object by taking vqa and vqaRes
	vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

	# evaluate results
	"""
	If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
	By default it uses all the question ids in annotation file
	"""
	vqaEval.evaluate() 

	# print accuracies
	print "\n"
	print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
	print "Per Question Type Accuracy is the following:"
	for quesType in vqaEval.accuracy['perQuestionType']:
		print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
	print "\n"
	print "Per Answer Type Accuracy is the following:"
	for ansType in vqaEval.accuracy['perAnswerType']:
		print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
	print "\n"
	# demo how to use evalQA to retrieve low score result
	evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
	if len(evals) > 0:
		print 'ground truth answers'
		randomEval = random.choice(evals)
		randomAnn = vqa.loadQA(randomEval)
		vqa.showQA(randomAnn)

		print '\n'
		print 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval])
		ann = vqaRes.loadQA(randomEval)[0]
		print "Answer:   %s\n" %(ann['answer'])

	# save evaluation results to ./Results folder
	json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
	json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
	json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
	json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

