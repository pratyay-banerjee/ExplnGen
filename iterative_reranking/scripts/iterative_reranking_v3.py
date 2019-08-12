#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import spacy
import nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm,tqdm_notebook
import string
import csv
import os
import logging
import argparse
import random
import pickle
import math
import sys
import warnings
from collections import namedtuple, OrderedDict
from functools import partial
from tqdm import tqdm
import pandas as pd
from pprint import pprint


print("Loading stopwords")
stpwords =  set(stopwords.words('english'))
stemmer = PorterStemmer()
print("Loading Spacy")
nlp = spacy.load("en_core_web_lg",disable=["ner"])
docmap={}
explanation_map={}
hypmap={}
hyp_exp_simscore={}
exp_exp_simscore={}


class ListShouldBeEmptyWarning(UserWarning):
    pass


Question = namedtuple('Question', 'id explanations')
Explanation = namedtuple('Explanation', 'id role')


def load_gold(filepath_or_buffer, sep='\t'):
    df = pd.read_csv(filepath_or_buffer, sep=sep)
    gold = OrderedDict()
    for _, row in df[['questionID', 'explanation']].dropna().iterrows():
        explanations = OrderedDict((uid.lower(), Explanation(uid.lower(), role))
                                   for e in row['explanation'].split()
                                   for uid, role in (e.split('|', 1),))
        question = Question(row['questionID'].lower(), explanations)
        gold[question.id] = question

    return gold


def load_pred(filepath_or_buffer, sep='\t'):
    df = pd.read_csv(filepath_or_buffer, sep=sep, names=('question', 'explanation'))
    pred = OrderedDict()
    for question_id, df_explanations in df.groupby('question'):
        pred[question_id.lower()] = list(OrderedDict.fromkeys(df_explanations['explanation'].str.lower()))

    return pred


def compute_ranks(true, pred):
    ranks = []
    if not true or not pred:
        return ranks
    targets = list(true)
    # I do not understand the corresponding block of the original Scala code.
    for i, pred_id in enumerate(pred):
        for true_id in targets:
            if pred_id == true_id:
                ranks.append(i + 1)
                targets.remove(pred_id)
                break

    # Example: Mercury_SC_416133
    if targets:
        warnings.warn('targets list should be empty, but it contains: ' + ', '.join(targets), ListShouldBeEmptyWarning)
        for _ in targets:
            ranks.append(0)
    return ranks


def average_precision(ranks):
    total = 0.
    if not ranks:
        return total
    for i, rank in enumerate(ranks):
        precision = float(i + 1) / float(rank) if rank > 0 else math.inf
        total += precision
    return total / len(ranks)


def mean_average_precision_score(gold, pred, callback=None):
    total, count = 0., 0
    for question in tqdm(gold.values()):
        if question.id in pred:
#             print(question.explanations)
            ranks = compute_ranks(list(question.explanations), pred[question.id])
            score = average_precision(ranks)
            if not math.isfinite(score):
                score = 0.

            total += score
            count += 1
            if callback:
                callback(question.id, score)

    mean_ap = total / count if count > 0 else 0.
    return mean_ap

def save_maps(fname,mmap):
    with open(fname, 'wb+') as handle:
        pickle.dump(mmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_doc_map(path_to_docmap):
    pickle_in = open(path_to_docmap,"rb")
    doc_map = pickle.load(pickle_in)
    pickle_in.close()
    return doc_map

def get_doc(docs):
    if docs in docmap:
        return docmap[docs]
    docmap[docs] = nlp(docs)
    return docmap[docs]


def load_explanations(path_to_explanations):
    explanations = pd.read_csv(path_to_explanations)
    explanation_map = {}
    for index,row in tqdm(explanations.iterrows(),desc="Loading Explanations:"):
        explanation_map[row['uid']] = row['text']
    return explanation_map

def load_simmap(source_map,explanation_map,src=None) :
    if src is not None:
        return load_doc_map(src)
    src_exp_simscore = {}
    for uid1,src in tqdm(source_map.items(),desc='Loading Source-Exp Map:'):
        srcdoc = get_doc(src)
        src_exp_simscore[src]={}
        for uid2,exp2 in explanation_map.items():
            exp2doc = get_doc(exp2)
            src_exp_simscore[src][exp2]=srcdoc.similarity(exp2doc)
    return src_exp_simscore

def get_score_map(df):
    q_uid_map = {}
    q_hyp_map = {}
    for index,row in tqdm(df.iterrows()):
        idx = row["id"]
        qid = idx.split(":")[0]
        uid = idx.split(":")[1]
        if qid not in q_uid_map:
            q_uid_map[qid]=[]
            q_hyp_map[qid] = row['hyp']
        q_uid_map[qid].append([uid,row['fact'],float(row['score'])])
    return q_uid_map,q_hyp_map

def rerank(hyp,tups,exp_exp_simscore,hyp_exp_simscore,topk=20,start=0):    
    result = {}
    total_len = len(tups)
    src = tups
    result=[]  # Put the first in result
    for ix in range(0,start):
        result.append(src[0])
        src.remove(src[0])
    curfact = src[start][1]
    cur = get_doc(src[start][1])
    cur_tup = src[start]
    information_map_n = {}
    information_map_d = {}
#     hypdoc = get_doc(hyp)
    prev = None
    ixx =0
    done = {}
    done[curfact]=True
    while src:
        if len(result) == topk:
            break
        ixx+=1
        temp = []
        for tup in src[start:start+topk]:
#             doc = get_doc(tup[1])
#             if done.get(tup[1],False):
#                 continue
            numerinfo = information_map_n.get(tup[1],0)
            denominfo = information_map_d.get(tup[1],0)
#             for selfact in result:
#                 fact = selfact[1]
#                 fact_score = selfact[2]
#                 similarity_with_tup = exp_exp_simscore[fact][tup[1]]
#                 numerinfo += similarity_with_tup*fact_score
#                 denominfo += fact_score
            similarity_with_tup = exp_exp_simscore[curfact][tup[1]]
            fact_score = cur_tup[2]
            numerinfo += similarity_with_tup*fact_score
            denominfo += fact_score
            information_map_n[tup[1]]=numerinfo
            information_map_d[tup[1]]=denominfo
            information = numerinfo/denominfo
            similarity = hyp_exp_simscore[hyp][tup[1]]
            temp.append([tup[0],tup[1],tup[2],tup[2]*information*similarity])
        if len(temp) == 0:
            break
        sorted_x = sorted(temp, key=operator.itemgetter(-1),reverse=True)
        tup = sorted_x[0]
        result.append(tup)
        if [tup[0],tup[1],tup[2]] in src:
            src.remove([tup[0],tup[1],tup[2]])
#         done[tup[1]]=True
        prev = cur
        curfact = tup[1]
        cur_tup = tup
        cur = get_doc(tup[1])
#     print(cur,prev)
    result.extend(src)
    return list(result)

def rerank_dataframe(scoremap,hypmap,exp_exp_simscore,hyp_exp_simscore,topk,start=0):
    ir_dataset_f2= {}
    ix=0
    for qidx,rows in tqdm(scoremap.items(),desc="Reranking:"):
        sorted_x = sorted(rows, key=operator.itemgetter(-1),reverse=True)
        rows = rerank(hypmap[qidx],list(sorted_x),exp_exp_simscore,hyp_exp_simscore,topk,start=0)
        ir_dataset_f2[qidx]=rows
    return ir_dataset_f2

def write_rerank_file(ir_map,fname):
    with open(fname+"-reranked.tsv","w+") as ofd:
        for uid,exps in tqdm(ir_map.items(),desc="Writing to File:"):
            for tup in exps:
                ofd.write("%s\t%s\n"%(uid,tup[0]))
    return fname+"-reranked.tsv"

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--fname",
                        default=None,
                        type=str,
                        required=True,
                        help="Source QuestionAnswer File")
    parser.add_argument("--scorefname", default=None, type=str, required=True,
                        help="Score File of source")
    parser.add_argument("--expfname", default=None, type=str, required=True,
                        help="Explanation File")
    parser.add_argument("--docmap", default=None, type=str, required=True,
                        help="PreComputed Spacy Docs")
    parser.add_argument('--gold', type=argparse.FileType('r', encoding='UTF-8'), required=False)
    parser.add_argument('--eval', action='store_true',
                        help="Whether to do eval param search.")
    parser.add_argument("--topk",default=15,type=int,help="Topk for prediction")
    
    
    args = parser.parse_args()
    print("Loading Doc Map")
    docmap = load_doc_map(args.docmap)
    scores = pd.read_csv(args.scorefname,delimiter="\t",names=["id","score"])
    orig = pd.read_csv(args.fname,delimiter="\t",names=["id","hyp","fact","label"])
    merged = pd.merge(scores, orig, on='id')
    
    explanation_map = load_explanations(args.expfname)
    
    print("Loading Explanation Similarity Map")
    exp_exp_simscore = load_simmap(explanation_map,explanation_map,"expexpmap.pickled")
    print(exp_exp_simscore['flexibility is a property of a material']['flexibility is a property of a material'])
    
    print("Loading Hypothesis and Scoring Map")
    
    if False:
        scoremap,hypmap =get_score_map(merged)
        save_maps("devscoremap.pickled",scoremap)
        save_maps("devhypmap.pickled",hypmap)
    elif args.eval:
        scoremap = load_doc_map("devscoremap.pickled")
        hypmap = load_doc_map("devhypmap.pickled")
    else:
        scoremap,hypmap =get_score_map(merged)
        
    print("Loading Hyp-Explanation Similarity Map")
    if args.eval:
        hyp_exp_simscore = load_simmap(hypmap,explanation_map,"devhypexpmap.pickled")
    else:
        hyp_exp_simscore = load_simmap(hypmap,explanation_map)
    
#     if True:
#         save_maps("expexpmap.pickled",exp_exp_simscore)
#         save_maps("devhypexpmap.pickled",hyp_exp_simscore)
    
    if args.eval:
        topkmap = {}
        gold = load_gold(args.gold)
        for topk in tqdm(range(16,17),desc="Running for Topk:"):
            ir_dataset_f2 = rerank_dataframe(scoremap,hypmap,exp_exp_simscore,hyp_exp_simscore,topk=topk)
            outfname = write_rerank_file(ir_dataset_f2,args.fname+str(topk))
            pred =  load_pred(outfname)
            # callback is optional, here it is used to print intermediate results to STDERR
            mean_ap = mean_average_precision_score(
                gold, pred,
#                 callback=partial(print, file=sys.stderr)
            )

            print('MAP: ', mean_ap)
            topkmap[topk] = mean_ap

        pprint(topkmap)
    else:
        topk = args.topk
        ir_dataset_f2 = rerank_dataframe(scoremap,hypmap,exp_exp_simscore,hyp_exp_simscore,topk)
        outfname = write_rerank_file(ir_dataset_f2,args.fname+str(topk))
    
    
if __name__ == "__main__":
    main()