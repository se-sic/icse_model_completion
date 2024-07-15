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

    return cleaned_parts


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
completion_generated_filtered = df_filtered['completion_string']
# We get the completion className and name  
df_filtered["classNames_generated"] = ""
df_filtered["names_generated"] = ""


#TODO also perform some stemming here


# We get ground truth className and name 
df_filtered[['classNames_ground_truth', 'names_ground_truth']] = completion_ground_truth_filtered.apply(
    lambda x: clean_and_split(x)).apply(pd.Series)

df_filtered[ ['classNames_generated', 'names_generated']] = completion_generated_filtered.apply(
    lambda x: clean_and_split(x)).apply(pd.Series)



# check when they are the same
same_className = df_filtered['classNames_ground_truth'] == df_filtered['classNames_generated']
print(same_className)
same_name = df_filtered['names_ground_truth'] == df_filtered['names_generated']
print(same_name)
print(f'Relative amount of completions where the className is the same: {same_className.sum() / len(same_className)}')
print(f'Absolute amount of completions where the className is the same: {same_className.sum()}')
print(f'Relative amount of completions where the name is the same: {same_name.sum() / len(same_name)}')
print(f'Absolute amount of completions where the name is the same: {same_name.sum()}')


# concate the same_name information to the original dataframe
df_filtered['same_name'] = same_name
df_filtered['same_class'] = same_className

# for all names that are not the same, print (on by one) the ground truth and generated name, and the full completion strings
for i, row in df_filtered.iterrows():
    if row['same_name'] == False:
        print(f'Ground truth completion: {row["completion"]}')
        print(f'Generated completion: {row["completion_string"]}')
        print('')
        
# save the dataset with the additional information
df_filtered.to_csv(output_path)



