#TODO <THIS IS DONE FOR DEBUGGING REASONS, if true only the first item is send to GPT
DEBUGGING_MODE = True
TEMPERATURE = 0.7 #like specified in the chaaben et al. baseline
MAX_TOKENS = 15 #like specified in the chaaben et al. baseline
MODEL_ID = "gpt-3.5-turbo-0125" #TO have a fairer comparison we use also gpt4
SYSTEM_INSTRUCTION = 'Continue the line: \n ' #like specified in the chaaben et al. baseline

####################################################################################################################
####################################################################################################################
####################################### baseline spezific implementation ###########################################
####################################################################################################################
####################################################################################################################
import os
from openai import OpenAI
from openai import Completion

# Load API key and setup OPEN-AI Lib
if not os.path.exists('../secrets/openai.key'):
    print("WARN: You need to provide your OpenAI API-Key in a file /secrets/openai.key")

with open('../secrets/openai.key', 'r') as f:
    api_key = f.read().strip()
    os.environ['OPENAI_API_KEY']=api_key

client = OpenAI()
import openai
import pandas as pd
import re
import wordninja 

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

    MODEL_ID = "davinci-002" 
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
    # Uncomment if you have these functions implemented
    # total_tokens, completion_tokens = token_counter_gpt(result)
    #concepts = retrieveConcepts(str(designList_) + ',' + completion_string)

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
    temperature=TEMPERATURE,        # Controls randomness in the output
    frequency_penalty=0,    # Controls the likelihood of repeating words/phrases
    presence_penalty=0      # Controls the likelihood of introducing new topics
    )

   # total_tokens, completion_tokens = token_counter_gpt(result)
    completion_string = completion_gpt_chat(result)
    #Wenn ich das vom paper hole, macht es auf unseren daten keinen sinn mehr
   # concepts = retrieveConcepts(str(designList_) + ',' + completion_string)

    
    return completion_string,  #total_tokens, completion_tokens




#read the context, so partical models from our revision dataset
input_path='./datasets_reduced/revision/results/baseline_data.csv'
output_path='./datasets_reduced/revision/results/results_baseline_chatgpt.csv'

context_chaaben =pd.read_csv(input_path)

#read the fewshots examples from baseline (Chaaben et al. paper) given in data.csv
few_shot_examples_chaaben =pd.read_csv('./datasets_reduced/BaselineFewShots/data.csv')
few_shot_examples_chaaben_data=''
for i, row in few_shot_examples_chaaben.iterrows():
  few_shot_examples_chaaben_data= few_shot_examples_chaaben_data+ '\n' + str(row['sequence']).strip('{}')+ '.'


# select data, i.e. fixed_prompt column. select only rows where completion_type = ['Add_node']
count = 0
for index, row in context_chaaben.iterrows():
    if row['completion_type'] == "['Add_node']" and re.match(r".*'changeType': 'Add', 'type': 'object'.*",row['completion']):
        count += 1

print(f"There are {count} rows with completion_type = ['Add_node']")


# select data, i.e. fixed_prompt column. select only rows where completion_type = ['Add_node']
count = 0
for index, row in context_chaaben.iterrows():
    if row['completion_type'] == "['Add_node']" and re.match(r".*'changeType': 'Add', 'type': 'object'.*",row['completion']):

        
        completion_string= predictFinalListDavinci(few_shot_examples_chaaben_data,  row['prompt'])

        count += 1
        # set in data file
        context_chaaben.at[index, 'completion_generated'] = completion_string
    #    context_chaaben.at[index, 'completion_tokens'] = completion_tokens
     #   context_chaaben.at[index, 'total_tokens'] = total_tokens
        # Save the DataFrame to CSV in each iteration
        context_chaaben.to_csv('results_baseline_chatgpt.csv', index=False)
        if DEBUGGING_MODE == True:
            break


      


# save the data
context_chaaben.to_csv(output_path, index=False)



