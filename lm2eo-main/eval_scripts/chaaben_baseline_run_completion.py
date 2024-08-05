import ast
import os
import numpy
from openai import OpenAI
from openai import Completion
import pandas as pd
import re
import wordninja 
import ast
from itertools import chain, combinations

# Load API key and setup OPEN-AI Lib
if not os.path.exists('../secrets/openai.key'):
    print("WARN: You need to provide your OpenAI API-Key in a file /secrets/openai.key")

with open('../secrets/openai.key', 'r') as f:
    api_key = f.read().strip()
    os.environ['OPENAI_API_KEY']=api_key

client = OpenAI()

#TODO <THIS IS DONE FOR DEBUGGING REASONS, if true only the first item is send to GPT
DEBUGGING_MODE = False
TEMPERATURE = 0.7 #like specified in the chaaben et al. baseline
MAX_TOKENS = 20 #like specified in the chaaben et al. baseline
MODEL_ID = "davinci-002"  #TO have a fairer comparison we use also gpt4
#SYSTEM_INSTRUCTION = 'Continue the line: \n ' #like specified in the chaaben et al. baseline
SYSTEM_INSTRUCTION = 'Generate related concepts: \n'

#read the context, so partical models from our revision dataset
input_path='./datasets_reduced/revision/results/baseline_data_withanchor.csv'
output_path='./datasets_reduced/revision/results/results_baseline_chatgpt_powerset_withanchor.csv'

####################################################################################################################
####################################################################################################################
####################################### baseline spezific implementation ###########################################
####################################################################################################################
####################################################################################################################


def completion_gpt_chat(result: Completion):
    return result.choices[0].message.content

def token_counter_gpt(result: Completion):
  # get token from result
  total_tokens = result.usage.total_tokens # completion_tokens
  completion_tokens = result.usage.completion_tokens
  return total_tokens, completion_tokens

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

def predictFinalListDavinci(datafewshots, designList_):
    prompt = SYSTEM_INSTRUCTION + datafewshots + '\n' + str(designList_) + ','

    result = client.completions.create(
            model=MODEL_ID,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            n=1,
            top_p=1,
            temperature=TEMPERATURE,  # Controls randomness in the output
            frequency_penalty=0,      # Controls the likelihood of repeating words/phrases
            presence_penalty=0        # Controls the likelihood of introducing new topics
        )
    
    completion_string = result.choices[0].text.strip()
    #We cant use this on our data, since they are slightly different 
    #also in the originial implementation this seems to be a bug, because
    #concepts = retrieveConcepts(str(designList_) + ',' + completion_string)
    #so the partical model is also appened
  

    return completion_string 

def predictFinalListChatModels(datafewshots, designList_):
    prompt =  datafewshots + '\n' + str(designList_) + ','

    result = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt}
    ], 
    max_tokens=MAX_TOKENS, 
    n=1, 
    top_p=1,  
    temperature=TEMPERATURE,# Controls randomness in the output
    frequency_penalty=0,    # Controls the likelihood of repeating words/phrases
    presence_penalty=0      # Controls the likelihood of introducing new topics
    )


    completion_string = completion_gpt_chat(result)
    return completion_string

def powerset(input_string):
    # Parse the string representation of the list of lists into an actual list of tuples
    input_list = ast.literal_eval(input_string)
    if type(input_list[0]) is not list:
        # It's a single list, so wrap it in another list to handle uniformly
        input_list = [input_list]
    # Helper function to generate all possible combinations of elements in a list
    def all_combinations(lst):
        return chain(*[combinations(lst, i) for i in range(1, len(lst)+1)])
    
    # Generate power set
    result = list(all_combinations(input_list))
    
    return result



context_chaaben =pd.read_csv(input_path)

#read the fewshots examples from baseline (Chaaben et al. paper) given in data.csv
few_shot_examples_chaaben =pd.read_csv('./datasets_reduced/BaselineFewShots/data.csv')
few_shot_examples_chaaben_data=''
for i, row in few_shot_examples_chaaben.iterrows():
  few_shot_examples_chaaben_data= few_shot_examples_chaaben_data+ '\n' + str(row['sequence']).strip('{}')+ '.'


# select data, i.e. fixed_prompt column. select only rows where completion_type = ['Add_node']
count = 0
for index, row in context_chaaben.iterrows():
    if row['completion_type'] == "['Add_node']" and re.match(r".*'changeType': 'Add', 'type': 'object'.*",row['ramc_completion']):
        count += 1

print(f"There are {count} rows with completion_type = ['Add_node']")


# select data, i.e. fixed_prompt column. select only rows where completion_type = ['Add_node']
count = 0
for index, row in context_chaaben.iterrows():
    if row['completion_type'] == "['Add_node']" and re.match(r".*'changeType': 'Add', 'type': 'object'.*",row['ramc_completion']):
        if (pd.isna(row['prompt'])): 
            continue
        all_prompts_combinations = powerset(row['prompt'])

        all_results =[]
        for item in all_prompts_combinations: 
            prompt = item[0] 
        
            completion_string= predictFinalListDavinci(few_shot_examples_chaaben_data,  str(prompt))
            all_results.append(completion_string)
            count += 1
        
        
        context_chaaben.at[index, 'completion_string'] = all_results

        # Save the DataFrame to CSV in each iteration
        context_chaaben.to_csv(output_path, index=False)

        if DEBUGGING_MODE == True:
            print("stopping")
            break
         
context_chaaben.to_csv(output_path, index=False)



