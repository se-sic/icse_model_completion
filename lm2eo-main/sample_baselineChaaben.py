

#!/usr/bin/python  

"""
    From given training and test set of single graph prompt completion pairs, this script filters them and creates few-shot samples, i.e., several training graphs and a test sample.
    
    There are two modes: 
        presample -> This filters the initial samples (e.g., based on token count of one sample) and then selects a certain fraction of the samples.
        finalsample -> This consumes the data from the presampling (probably a manual curation takes place in-between), and then create few-shot and test prompts and some additional meta-info.
    
    The presampling is thought to be used too downsample the data for a manual curation.
    After manual curation of the dataset, finalsampling can be used
    
    Call as python ./sample.py [presample|finalsample] ./../model_completion_dataset/Siemens_Mobility/results/experiment_samples/ ./../model_completion_dataset/Siemens_Mobility/results/experiment_selected_prompts/
    
"""

import ast
import json
import math
import re
import sys, os
from typing import List
import pandas as pd
import random
import logging
from edgel_utils import get_prompt_graphs, parse_edge, V_JSON, clean_up_string
from vector_database import ChangeGraphVectorDB
from sample import select_test_samples, select_few_shot_samples_for_test_samples
script_dir = os.path.dirname(os.path.abspath(__file__))
# TODO write sampling results (e.g., number of removed samples etc) to a csv-file (extend the one from the dataset generation step)



#TODO call sample.py  finalsample ./datasets_reduced/revision/vector_database_new/  ./datasets_reduced/revision/final_few_shot_dataset_baseline/ with RANDOM!
#Richtig : finalsample ./datasets_reduced/revision/vector_database_new/  ./datasets_reduced/SimpleComponentModel/final_few_shot_dataset_baseline/



#utput_path= './datasets_reduced/revision/final_few_shot_dataset_baseline/'
#df = pd.read_json(output_path  + '/few_shot_dataset.jsonl', lines=True)
output_path='./datasets_reduced/revision/results/results_gpt_completion.csv'
df = pd.read_csv(output_path)
#please filter the csv where 
# completion_type == ['Add_node'] and completion.str.contains(''changeType': 'Add', 'type':'object'")


#filtered_df = df[df['completion_type'].str.contains("['Add_node']")]
filtered_df = df[df['completion_type'].str.contains("Add_node")]
filtered_df = filtered_df[filtered_df['completion'].str.contains("'changeType': 'Add', 'type': 'object'")]


def get_name(data_dict):
        attributes = data_dict.get('attributes', {})
        if 'name' in attributes:
            return attributes['name']
        elif 'key' in attributes:
            return attributes['key']
        elif 'eModelElement' in attributes:
            return attributes['eModelElement']
        else:
            return ''
        
def get_corresponding_names(data_dict_source_node,data_dict_target_node ):

    source_name = get_name(data_dict_source_node)
    target_name = get_name(data_dict_target_node)
    return f"[{source_name},{target_name}]"

def parse_last_entry(entry):
        entries= get_prompt_graphs(entry)
       # entries = entry.split('$$\n---\n')
        last_entry = entries[-1].strip()  # Get the last entry and strip whitespace
        edge_dict_format={}
        final_baseline_info =""
        # Parse the JSON-like string into a dictionary
        if last_entry:
            lines = last_entry.split('\n')
            lines=lines[1:] #is this correct to cut away first elemtn?

            for edge in lines: 
          

                is_edge, src_id, tgt_id, edge_label, src_node_label, tgt_node_label = parse_edge(edge, version=V_JSON)
                if (not is_edge): 
                    print("problem with reading string")
                    break; 

                if src_node_label == '"{}"':
                    src_node_label = edge_dict_format[src_id]

                if tgt_node_label == '"{}"':
                    tgt_node_label = edge_dict_format[tgt_id]

                edge_dict_format[src_id] = src_node_label
                edge_dict_format[tgt_id] = tgt_node_label

                try:
                    src_node_label_info = json.loads(clean_up_string(src_node_label))
                    data_dict_source_node = ast.literal_eval(src_node_label_info)
                
                except (json.JSONDecodeError, ValueError) as e:
                    print("Failed to decode JSON for source node:", e)
                    data_dict_source_node = {}
                try:
                    tgt_node_label_info = json.loads(clean_up_string(tgt_node_label))
                    data_dict_target_node = ast.literal_eval(tgt_node_label_info)
                except (json.JSONDecodeError, ValueError) as e:
                    print("Failed to decode JSON for target node:", e)
                    data_dict_target_node = {}

                if data_dict_source_node and data_dict_target_node:
                    try:
                        final_baseline_info += get_corresponding_names(data_dict_source_node, data_dict_target_node)
                    except KeyError as e:
                        print("Key error while accessing node attributes:", e)

        return final_baseline_info
          #  final_baseline_info+= "["+ data_dict_source_node['attributes']['name'] + ","+ data_dict_target_node['attributes']['name'] +"]"


    
df_partical_mode_to_complete=[]
df_partical_mode_to_complete.extend(filtered_df["prompt"].apply(parse_last_entry).tolist())


#read the fewshots examples from baseline given in data.csv
few_shot_examples_chaaben =pd.read_csv('../dataset_chaaben/data.csv')

few_shot_examples_chaaben_data=''
for i, row in few_shot_examples_chaaben.iterrows():
  few_shot_examples_chaaben_data= few_shot_examples_chaaben_data+ '\n' + str(row['sequence']).strip('{}')+ '.'







####################################################################################################################
####################################################################################################################
####################################### baseline spezific impl######################################################
####################################################################################################################
####################################################################################################################
import openai
import wordninja


def removeDuplicated(TheList):
    for ind, j in enumerate(TheList):
        for indx, i in enumerate(TheList[ind + 1:]):
            if i.lower() == j.lower():
                TheList.remove(j)
    return TheList

def removeLetter(TheList):
    for ind, j in enumerate(TheList):

        if len(j) == 1:
            TheList.remove(j)
    return TheList

def retrieveConcepts(res):
    S = re.sub(r'\W+', ' ', res)
    FirstSet = S.split(' ')
    NewConcepts = wordninja.split(S)
    NewConcepts.extend(FirstSet)
    NewConcepts = ' '.join(NewConcepts).split()
    concepts = set(NewConcepts)
    concepts = list(concepts)
    concepts = removeLetter(concepts)
    concepts = removeDuplicated(concepts)

    return concepts
def predictFinalList(designList_):
    Prompt = 'Continue the line: \n ' + data + '\n' + str(designList_) + ','

    response_ = openai.Completion.create(
        engine="text-davinci-002",
        prompt=Prompt,
        temperature=0.7,
        max_tokens=15,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    res_ = response_.choices[0].text
    concepts = retrieveConcepts(str(designList_) + ',' + res_)

    return concepts, res_


