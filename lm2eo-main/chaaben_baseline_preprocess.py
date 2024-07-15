

#!/usr/bin/python  

import ast
import json
import  os
from typing import List
import pandas as pd

from edgel_utils import get_prompt_graphs, parse_edge, V_JSON, clean_up_string
script_dir = os.path.dirname(os.path.abspath(__file__))
# TODO write sampling results (e.g., number of removed samples etc) to a csv-file (extend the one from the dataset generation step)




input_path='./datasets_reduced/revision/results/results_gpt_completion.csv'
df = pd.read_csv(input_path)


# List of columns to be removed
columns_to_remove = [
     "few_shot_token_count", "test_token_count", "total_token_count",
    "few_shot_count", "test_edge_count", "test_completion_edge_count", "correct_format",
    "type_isomorphic_completion", "change_type_isomorphic_completion", "structural_isomorphic_completion",
    "completion_string", "completion_tokens", "total_tokens"
]

# Remove the columns if they exist in the DataFrame
df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

filtered_df = df[df['completion_type'].str.contains("Add_node")]
filtered_df = filtered_df[filtered_df['completion'].str.contains("'changeType': 'Add', 'type': 'object'")]


def get_name(data_dict):
        class_name = data_dict.get('className', {}) +"."
        attributes = data_dict.get('attributes', {})
        if 'name' in attributes:
            return class_name + attributes['name']
        elif 'key' in attributes:
            return class_name+ attributes['key']
        elif 'eModelElement' in attributes:
            return class_name+ attributes['eModelElement']
        else:
            return
        
def get_corresponding_names(data_dict_source_node,data_dict_target_node ):

    source_name = get_name(data_dict_source_node)
    target_name = get_name(data_dict_target_node)
    return f"['{source_name}','{target_name}']"

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
                        if final_baseline_info=="":
                             final_baseline_info += get_corresponding_names(data_dict_source_node, data_dict_target_node)
                        else: 
                            
                            final_baseline_info += ","+get_corresponding_names(data_dict_source_node, data_dict_target_node)
                    except KeyError as e:
                        print("Key error while accessing node attributes:", e)
                    
      
        return final_baseline_info
          #  final_baseline_info+= "["+ data_dict_source_node['attributes']['name'] + ","+ data_dict_target_node['attributes']['name'] +"]"



#TODO spalte ground_truth_baseline_transformed

filtered_df.rename(columns={"prompt": "ramc_prompt"}, inplace=True)
filtered_df.rename(columns={"completion": "ramc_completion"}, inplace=True)


filtered_df["prompt"] = filtered_df.apply(lambda row: parse_last_entry(row["ramc_prompt"]), axis=1)


filtered_df.loc[filtered_df.index != filtered_df.index, "prompt"] = ""


df["prompt"] = ""


# Update the original DataFrame with values from the filtered DataFrame
df.update(filtered_df[["prompt"]])

# Save the modified DataFrame

output_path='./datasets_reduced/revision/results/baseline_data.csv'
df.to_csv(output_path, index=False)






