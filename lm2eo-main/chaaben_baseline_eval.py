import ast
import pandas as pd
import re
from collections import Counter
output_path ='./datasets_reduced/revision/results/dataset_chaaben_baseline_with_completions.csv'
# Given a csv file with the above format, this script will extract the completion and the completion_string from the csv
dataset = pd.read_csv('./datasets_reduced/revision/results/results_baseline_chatgpt_powerset_withanchor.csv')




#TODO WE NEED TO ADD STH HERE
#TODO get_name_or_key could be wrong
def get_name_or_key(full_string: str, identifier: str = ['name', 'key']):
    # regex for every identifier in the identifier list "'({identifier1}|{identifier2}|....})':'(\w+)'"
    regex = f"'({'|'.join(identifier)})':'(\w+)'"
    # map to regex
    matches = re.findall(regex, full_string)
    # get second group 
    assert len(matches) > 0, f'No matches found for {identifier} in {full_string}'
    assert len(matches) < 2, f'More than one match found for {identifier} in {full_string}'
    return matches[0][1]

def get_simple_name(class_name: str | float | None, name: float | str | None):
    if class_name is None or name is None or type(name) == float or type(class_name) == float:
        return None

def find_most_frequent(list_recommendations):
    #never the case if len(list_recommendations)==0: 
       # return ("",0)
    # Count the occurrences of each string in the list
    count = Counter(list_recommendations)
    # Find the string with the highest frequency
    if len(count)==0: 
        return ("",0)
    most_frequent = count.most_common(1)[0]  # This returns a tuple (element, count)
    #inever the casef most_frequent=="" and len(count) > 1:
      #  most_frequent= count.most_common(2)[1]

    return most_frequent   
    return class_name + '.' + name
def rank_for_connection(input_str):
    if (input_str==''): 
        #this is just import for that one example
        return 'x'
    inputs = ast.literal_eval(input_str)
    list_recommendations = [clean_and_split(i)for i in inputs]
    most_frequent_string = find_most_frequent(list_recommendations)
    return most_frequent_string[0]

def split_concept(concept):
    # Split the concept at the period
    parts = concept.split('.')
    
    # Ensure there are exactly two parts; if not, handle it by providing a default
    if len(parts) == 1:
        # Assuming the concept should at least have a class name, append an empty string for the name
        parts.append('')  # Default or missing name
    elif len(parts) > 2:
        # If there are more than two parts, this handles cases like 'EClass.Provides.Predicate'
        # You might want to join them back or handle differently depending on your data
        return parts[0], '.'.join(parts[1:])  # This keeps the first part, and joins the rest as the second part
    return parts[0], parts[1]

#the most likely assozation [. ] tuple
def rank_for_concepts_improved(row):
    simplified_gen_association = row['simplified_gen_association']
    anchor_node= row['anchor_node_gt']
    # Extract the string and convert it from string representation of list to actual list
    try:
        actual_list = ast.literal_eval(simplified_gen_association)
        if (actual_list[0]==anchor_node): 
            return actual_list[1]
        else: 
            return actual_list[0]
        
    except Exception as e:
        print("Error:", str(e))
        return 'x.x'

def rank_for_concepts_like_chaaben_but_no_targets(row): 
    input_str = row['completion_string']
    anchor_node = row['anchor_node_gt']
    if (input_str==''): 
        #this is just import for that one example, dcannot be empty becuase otherwise it matches gt 
        return 'x'
    inputs = ast.literal_eval(input_str)
    list_recommendations = [clean_and_split(i)for i in inputs]
    all_concepts = []
   
    for item in list_recommendations:
        if (item==''): 
            continue

        try:
            actual_list = ast.literal_eval(item)
            if (actual_list[0]!=anchor_node): 
                all_concepts.append(actual_list[0])
            if (actual_list[1]!=anchor_node): 
                all_concepts.append(actual_list[1])   
            if (actual_list[1]==anchor_node and actual_list[0]==anchor_node): 
                all_concepts.append(actual_list[1])   
         
        except Exception as e:
            print("Error:", str(e))

            continue

        # Convert the string representation of the list to an actual list
        
        # Append the first and second item from each list to their respective lists
      

    most_frequent_string = find_most_frequent(all_concepts)
    return most_frequent_string[0]


def rank_for_concepts_like_chaaben(input_str): 
    if (input_str==''): 
        #this is just import for that one example, dcannot be empty becuase otherwise it matches gt 
        return 'x'
    inputs = ast.literal_eval(input_str)
    list_recommendations = [clean_and_split(i)for i in inputs]
    all_concepts = []
   
    for item in list_recommendations:
        if (item==''): 
            continue

        try:
            actual_list = ast.literal_eval(item)
            all_concepts.append(actual_list[0])
            all_concepts.append(actual_list[1])   
         
        except Exception as e:
            print("Error:", str(e))

            continue

        # Convert the string representation of the list to an actual list
        
        # Append the first and second item from each list to their respective lists
      

    most_frequent_string = find_most_frequent(all_concepts)
    return most_frequent_string[0]



def split_after_first_bracket(input_str):
    # Split the string immediately after the first ']'
    parts = re.split(r'(?<=\])', input_str, 1)  # The '1' makes the split happen only once
    return parts

def clean_and_split(input_str):
    #ast.literal_eval(input_str)
    # Remove whitespace and newlines for easier processing
    input_str = input_str.replace(' ', '').replace('\n', '')

    # Split the string right after '],' but keep ']' in the parts
    parts  = split_after_first_bracket(input_str)

    cleaned_parts = []
    for part in parts:
        if not part.endswith(']'):
            # If last part doesn't end with ']', it's considered incomplete and skipped
            continue
        cleaned_parts.append(part)
    #just get the first element since we do not have more

    return cleaned_parts[0] if cleaned_parts else ""


completion_ground_truth_ramc= dataset['ramc_completion']

# next if available, we check how often the completion is of 'changeType': 'Add'  and 'type': 'object' (i.e., ``adding a new class'').
# Check how often the completion is of 'changeType': 'Add' and 'type': 'object' (i.e., adding a new class)
is_add_object = completion_ground_truth_ramc.str.contains("'changeType': 'Add', 'type': 'object'")

# Calculate relative and absolute amounts
relative_amount = is_add_object.sum() / len(completion_ground_truth_ramc)
absolute_amount = is_add_object.sum()

#print(f'Relative amount of completions that are of type 'Add object': {relative_amount}')
#print(f'Absolute amount of completions that are of type 'Add object': {absolute_amount}')

# Select only the add object case

df_filtered =dataset[is_add_object]

df_filtered['completion'] = df_filtered['completion'].fillna("")
df_filtered['simplified_gt_association'] = df_filtered ['completion']


df_filtered['completion_string'] = df_filtered['completion_string'].fillna("")

df_filtered[ 'simplified_gen_association'] = df_filtered['completion_string'].apply(rank_for_connection).apply(pd.Series)
#df_filtered[ 'simplified_concept_ranked'] = df_filtered['completion_string'].apply(rank_for_concepts_improved).apply(pd.Series)


#df_filtered['simplified_concept_ranked'] = df_filtered.apply(rank_for_concepts_improved, axis=1).apply(pd.Series)
df_filtered['simplified_concept_ranked'] = df_filtered.apply(rank_for_concepts_like_chaaben_but_no_targets, axis=1).apply(pd.Series)



# check when they are the same
df_filtered[ 'same_association'] = df_filtered['simplified_gt_association'] == df_filtered['simplified_gen_association']

df_filtered['same_concept'] = df_filtered['simplified_concept_ranked'] == df_filtered['new_node_gt']
# 2. Split the 'new_node_gt' column and assign to new columns
df_filtered[['class_names_gt', 'names_gt']] = df_filtered['new_node_gt'].str.split('.', expand=True)

# 3. Split the 'simplified_concept_ranked' column and assign to new columns
df_filtered[['class_names_gen', 'names_gen']] = df_filtered[['class_names_gen', 'names_gen']] = df_filtered['simplified_concept_ranked'].apply(lambda x: pd.Series(split_concept(x)))

df_filtered['same_class'] = df_filtered['class_names_gt'] == df_filtered['class_names_gen']

# Compare 'names_gt' and 'names_gen' for equality
df_filtered['same_name'] = df_filtered['names_gt'] == df_filtered['names_gen']

# Renaming the columns
df_filtered.rename(columns={'completion': 'chaaben_completion', 'ramc_completion': 'completion'}, inplace=True)

dataset.rename(columns={'completion': 'chaaben_completion', 'ramc_completion': 'completion'}, inplace=True)

for column in df_filtered.columns:
    dataset.loc[df_filtered.index, column] = df_filtered[column]
dataset['simplified_correct']=dataset[ 'same_association']
# save the dataset with the additional information
dataset.to_csv(output_path)




