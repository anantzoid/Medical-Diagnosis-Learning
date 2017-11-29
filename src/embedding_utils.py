from build_datasets_utils import *

def convert_flatdata_to_starspace_format(data, flag_labeled = True, label_prefix = "__label__"):
    ## expecting data after converted by build_datasets_utils.FlatData
    ## dict_keys(['text_index_sequence', 'label', 'id', 'text'])
    ## where 'label' is actually an index.
    
    output = []
    if flag_labeled == True:
        for i in range(len(data)):
            ## add label_prefix (__label__) to each label and add to list of words.
            ## join by space
            output.append(" ".join([label_prefix + d for d in data.get_dx_index(i)] + data.get_words(i)))
        print("Finished converting labeled data to StarSpace format.")
        return output
        ## Done
    elif flag_labeled == False:
        for i, rec in enumerate(data):
            ## ONLY list of words.
            ## join by space
            output.append(" ".join(data.get_words(i)))
        print("Finished converting UNlabeled data to StarSpace format.")
        return output
    
def write_starspace_format(records, filepath):
    with open(filepath, 'w') as file_handler:
        for record in records:
            file_handler.write("{}\n".format(record))
    print("Finished writing StarSpace data to file.")
    
    