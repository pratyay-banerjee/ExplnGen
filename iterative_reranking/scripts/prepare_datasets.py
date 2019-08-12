
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


# In[2]:


import nltk
from nltk.corpus import stopwords
stpwords =  set(stopwords.words('english'))


# In[3]:


import spacy
nlp = spacy.load("en_core_web_lg",disable=["ner","tagger","parser"])


# In[4]:


def get_correct_option_only(text,corr_option):
#     text = "Which describes a material that is not a food? (A) It stores energy but not nutrients. (B) It does not store energy or nutrients. (C) It stores energy and nutrients. (D) It does not store energy but stores nutrients."
    text_question = text.split("(A)")[0].strip()
    try:
        correct_option = text.split("("+corr_option+")")[1].split("(")[0]
    except:
        print(text)
#     print(text_question,correct_option)
    return text_question,correct_option


# In[5]:


def get_explanations(exps):
#     exps = 'eccc-ccef-2b84-af94|CENTRAL dd64-2412-f733-7659|LEXGLUE 0456-0148-2518-a48e|LEXGLUE 004d-1843-0f4a-8ad4|LEXGLUE b964-cdbd-a226-4027|CENTRAL 1ab8-7900-02c0-a2d6|CENTRAL d7c1-cdfd-2de5-6b7d|CENTRAL 1015-4572-2293-d185|LEXGLUE 0d3a-8739-9f42-5e1c|GROUNDING f6bf-c88d-502b-826b|CENTRAL 10ce-c060-90b9-b748|GROUNDING ff68-4699-0671-f8b7|LEXGLUE 7a74-426d-ec77-f296|LEXGLUE	'
    try:
        exps = exps.split(" ")
    except:
#         print(exps)
        pass
    
    out_list = [ x.split("|")[0].strip() for x in exps ]
#     print(out_list)
    return out_list

def get_roles(exps):
#     exps = 'eccc-ccef-2b84-af94|CENTRAL dd64-2412-f733-7659|LEXGLUE 0456-0148-2518-a48e|LEXGLUE 004d-1843-0f4a-8ad4|LEXGLUE b964-cdbd-a226-4027|CENTRAL 1ab8-7900-02c0-a2d6|CENTRAL d7c1-cdfd-2de5-6b7d|CENTRAL 1015-4572-2293-d185|LEXGLUE 0d3a-8739-9f42-5e1c|GROUNDING f6bf-c88d-502b-826b|CENTRAL 10ce-c060-90b9-b748|GROUNDING ff68-4699-0671-f8b7|LEXGLUE 7a74-426d-ec77-f296|LEXGLUE	'
    if "|" in exps:
        try:
            exps = exps.split(" ")
        except:
#             print(exps)
            pass

        out_list = [ x.split("|")[1].strip() for x in exps ]
    #     print(out_list)
        return out_list
    else:
        try:
            exps = exps.split(" ")
        except:
#             print(exps)
            pass
        return ["CENTRAL"]*len(exps)

def get_text_explanations(exps):
    outlist = []
    for x in exps:
        try: 
            text = explanation_map[x]
            outlist.append(text)
        except:
            print("Not Found:",x)
    return outlist

# get_text_explanations(get_explanations("a"))


# In[6]:


train = pd.read_csv("../train.csv",)
dev = pd.read_csv("../dev.csv")
test = pd.read_csv("../test.csv")


# In[7]:


def test_nan(df):
    print("Total Length:",len(df))
    print(df[df.isna().any(axis=1)].count())


# In[8]:


test_nan(train),test_nan(dev),test_nan(test)


# In[9]:


# Note : Train and Dev have few datapoints with no explanations. Ignore them


# In[10]:


explanations = pd.read_csv("../explanations.csv")


# In[11]:


len(explanations)


# In[12]:


explanations.head()


# In[13]:


explanation_map = {}
for index,row in tqdm(explanations.iterrows()):
    explanation_map[row['uid']] = row['text']


# In[ ]:


explanation_map['12d9-60df-23de-f00f']


# In[15]:


docmap= {}


# In[17]:


from random import shuffle
from itertools import cycle, islice
import operator

def strip_punct(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def load_doc_map(path_to_docmap):
    pickle_in = open(path_to_docmap,"rb")
    doc_map = pickle.load(pickle_in)
    pickle_in.close()
    return doc_map

doc_map = load_doc_map("docmap.pickled")

def get_doc(docs):
    if docs in docmap:
        return docmap[docs]
    docmap[docs] = nlp(docs)
    return docmap[docs]

def get_other_facts(factlist,l=50):
    wronglist = []
    explist = [x for x in explanation_map.items()]
    shuffle(explist)
    scored_wfact_map = {}
    for uid,wfact in explist:
        present = False
        wdoc = get_doc(wfact)
        score = -1
        for fact in factlist:
            if fact == wfact:
                present = True
                break
            fdoc =get_doc(fact)
            score = max(score,fdoc.similarity(wdoc))
        if not present:
            scored_wfact_map[uid]=score
    
    sorted_x = list(sorted(scored_wfact_map.items(), key=operator.itemgetter(1),reverse=True))[0:l]
    for tup in sorted_x:
        if tup[0] not in wronglist:
            wronglist.append([tup[0],explanation_map[tup[0]]])
    return wronglist

def get_other_fact_v2(fact,l=20):
    wronglist = []
    explist = [x for x in explanation_map.items()]
    shuffle(explist)
    scored_wfact_map = {}
    score=-1
    for uid,wfact in explist:
        present = False
        wdoc = get_doc(wfact)
        score = max(score,fdoc.similarity(wdoc))
        if not present:
            scored_wfact_map[uid]=score
            
    sorted_x = list(sorted(scored_wfact_map.items(), key=operator.itemgetter(1),reverse=True))[0:l]
    for tup in sorted_x:
        if [tup[0],explanation_map[tup[0]]] not in wronglist:
            wronglist.append([tup[0],explanation_map[tup[0]]])
    return wronglist
        
def get_same_facts(factlist,l=50):
    return list(islice(cycle(factlist), l))

print(get_other_facts(['earthworms create tunnels in soil',
 'to create means to make',
 'a tunnel is a kind of hole',
 'soil is a part of the ground outside',
 'tunnels in soil loosen that soil',
 'the looseness of soil increases the amount of oxygen in that soil',
 'plants absorb nutrients; water; oxygen from soil into themselves through their roots',
 'taking in something means receiving; absorbing; getting something',
 'a tree is a kind of plant',
 'an earthworm is a kind of animal',
 'an animal is a kind of organism',
 'to make something easier means to help',
 'getting something increases the amount of that something']))
            


# In[18]:


print(dev.head())


# In[123]:


# Prepare a Ranking Classification Dataset

def create_2_class_dataset(df,fname):
    lengthmap= {}
    with open("mod_data3/"+fname,"w+") as outfd:
        df_t = df.dropna()
        for index,row in tqdm(df_t.iterrows()):
            qid = row['questionID']
            cor_opt = row['AnswerKey']
            explanations = get_explanations(row['explanation'])
            exp_texts = get_text_explanations(explanations)
            ques,ans = get_correct_option_only(row['Question'],cor_opt)
            wrong_facts = get_other_facts(exp_texts)
            hyp = ques + " " + ans #mod_data1 and 2- 2 was better
#             hyp = row['Question']  # mod_data3
            for uid,fact in get_same_facts(list(zip(explanations,exp_texts))):
                idx = qid+":"+uid
                outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,1))
            for uid,fact in wrong_facts:
                idx = qid+":"+uid
                outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,0))
            if len(explanations) not in lengthmap:
                lengthmap[len(explanations)]=0
            lengthmap[len(explanations)]+=1
    print(lengthmap)


# In[133]:


def create_n_class_dataset(df,fname):
    lengthmap= {}
    with open("mod_datan/"+fname,"w+") as outfd:
        df_t = df.dropna()
        for index,row in tqdm(df_t.iterrows()):
            qid = row['questionID']
            cor_opt = row['AnswerKey']
            explanations = get_explanations(row['explanation'])
            exp_texts = get_text_explanations(explanations)
            ques,ans = get_correct_option_only(row['Question'],cor_opt)
            wrong_facts = get_other_facts(exp_texts)
            hyp = ques + " " + ans # mod_data1 and 2
#             hyp = row['Question']  # mod_data3
# Initial length
            for uid,fact in get_same_facts(list(zip(explanations,exp_texts))):
                idx = qid+":"+uid
                outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,1))
            for uid,fact in wrong_facts:
                idx = qid+":"+uid
                outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,0))
            if len(explanations) not in lengthmap:
                lengthmap[len(explanations)]=0
# Consequent length                
            for index,exp in enumerate(exp_texts):
                hyp = hyp + " . " + exp
                try:
                    wrong_facts = get_other_facts(exp_texts[index+1:])
                    for uid,fact in get_same_facts(list(zip(explanations,exp_texts[index+1:]))):
                        idx = qid+":"+uid+":"+str(index)
                        outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,1))
                    for uid,fact in wrong_facts:
                        idx = qid+":"+uid+":"+str(index)
                        outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,0))
                except:
                    pass
                
            lengthmap[len(explanations)]+=1
    print(lengthmap)


# In[21]:


roles_map = {
    'CENTRAL' : 1,
    'LEXGLUE' : 2,
    'GROUNDING' : 3,
    'ROLE' : 1,
    'NEG' : 1,
    'BACKGROUND' : 1,
}

def create_n_class_multi_dataset(df,fname):
    lengthmap= {}
    with open("mod_datamulti/"+fname,"w+") as outfd:
        df_t = df.dropna()
        for index,row in tqdm(df_t.iterrows()):
            qid = row['questionID']
            cor_opt = row['AnswerKey']
            explanations = get_explanations(row['explanation'])
            exp_texts = get_text_explanations(explanations)
            roles = get_roles(row['explanation'])
#             print(roles)
            ques,ans = get_correct_option_only(row['Question'],cor_opt)
            wrong_facts = get_other_facts(exp_texts,40)
            hyp = ques + " " + ans # mod_data1 and 2
#             hyp = row['Question']  # mod_data3
# Initial length
            for uid,fact,role in get_same_facts(list(zip(explanations,exp_texts,roles)),120):
                idx = qid+":"+uid
                outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,roles_map.get(role,1)))
            for uid,fact in wrong_facts:
                idx = qid+":"+uid
                outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,0))
            if len(explanations) not in lengthmap:
                lengthmap[len(explanations)]=0
# Consequent length                
            for index,exp in enumerate(exp_texts):
                hyp = hyp + " . " + exp
                try:
                    wrong_facts = get_other_facts(exp_texts[index+1:])
                    for uid,fact,role in get_same_facts(list(zip(explanations,exp_texts[index+1:],roles[index+1:])),120):
                        idx = qid+":"+uid+":"+str(index)
                        outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,roles_map.get(role,1)))
                    for uid,fact in wrong_facts:
                        idx = qid+":"+uid+":"+str(index)
                        outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,0))
                except:
                    pass
                
            lengthmap[len(explanations)]+=1
    print(lengthmap)


# In[125]:


def create_2_class_dataset_test(df,fname):
    with open("mod_data3/"+fname,"w+") as outfd:
        for index,row in tqdm(df.iterrows()):
            qid = row['questionID']
            cor_opt = row['AnswerKey']
#             ques,ans = get_correct_option_only(row['Question'],cor_opt)
#             hyp = ques + " " + ans
            hyp = row['Question']
            for uid,fact in explanation_map.items():
                idx = qid+":"+uid
                outfd.write("%s\t%s\t%s\t%d\n"%(idx,hyp,fact,0))


# In[25]:


create_n_class_multi_dataset(dev,"dev.tsv")


# In[ ]:


create_n_class_multi_dataset(train,"train.tsv")

