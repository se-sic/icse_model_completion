import pandas as pd
import re

output_path ='./datasets_reduced/revision/results/results_iso_check_with_info_baseline.csv'
# Given a csv file with the above format, this script will extract the completion and the completion_string from the csv
dataset = pd.read_csv('./datasets_reduced/revision/results/results_baseline_chatgpt.csv')




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
    
    return class_name + '.' + name

def clean_and_split(input_str):
    
    # Remove whitespace and newlines for easier processing
    input_str = input_str.replace(' ', '').replace('\n', '')

    # Split the string right after '],' but keep ']' in the parts
    parts = re.split(r'(?<=]),', input_str)

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

completion_ground_truth_filtered = df_filtered ['completion']
df_filtered['completion_string'] = df_filtered['completion_string'].fillna("")
completion_generated_filtered = df_filtered['completion_string']
# We get the completion className and name  


#TODO also perform some stemming here


# We get ground truth className and name 

df_filtered['simplified_gt'] = completion_ground_truth_filtered.apply(clean_and_split).apply(pd.Series)

df_filtered[ 'simplified_gen'] = completion_generated_filtered.apply(clean_and_split).apply(pd.Series)



# check when they are the same
same_all = df_filtered['simplified_gt'] == df_filtered['simplified_gen']
#print(same_className)
#same_name = df_filtered['names_ground_truth'] == df_filtered['names_generated']
print(same_all)
# save the dataset with the additional information
df_filtered.to_csv(output_path)



