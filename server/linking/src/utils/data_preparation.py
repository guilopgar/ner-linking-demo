import os, sys

def check_path_ending(input_dir):
    """Function to check the ending of the input path. if the path doesn't end with '/', It adds it to the end of it.
    :param input_dir: Path to the folder.
    :return: Complete directory
    """
    if input_dir[-1] != os.path.sep:
        output_dir = input_dir + os.path.sep
    else:
        output_dir = input_dir
    return output_dir

def texts2dict(path):
    """ Auxiliary funtion for reading txt files from path.
    :param path: Path to txt files
    :return: Dictionary
    """
    out_dict = dict()
    path = check_path_ending(path)
    for txt_file in os.listdir(path):
        out_dict[txt_file.split(".txt")[0]] = file_reader(path+txt_file)
    return out_dict

def file_reader(filename):
    """Auxiliary function for reading .txt files.
    :param filename: Path to txt file.
    :return: String variable with the file content.
    """
    return open(filename).read()


# Build paths to full text
def get_context_tokens(document, m_ini, m_end, additional_tokens):
    """Get tokens around mention inside a document given the initial
    and end offset number of the  mention. 

    Args:
        document (str): Document where mention appears
        m_ini (int): initial offset of the mention
        m_end (int): End offset of the mention
        additional_tokens (int): Number of tokens to take around the mention

    Returns:
        str: String output
    """
    # Initial and end token index in document for the mention
    mencion_tok_ini_n = len(document[:int(m_ini)].split())
    mencion_tok_end_n = len(document[:int(m_end)].split())
    # Get last initial and end token from the document 
    ini_tok_ind = 0 if mencion_tok_ini_n-additional_tokens<0 else mencion_tok_ini_n-additional_tokens
    ini_tok_end = mencion_tok_end_n+additional_tokens
    # Get out tokens
    out_tokens = document.split()[ini_tok_ind:ini_tok_end]
    # Join string
    return " ".join(out_tokens)




def transform_candidates_output_to_tuple(candidate_object):
    """Function that, given a candidate generation object, returns a list of prediction lists in 
    which each element is composed of the tuple ("code", "similarity").

    Args:
        candidate_object (candidates object): Candidate object

    Returns:
        list: List of lists of tuples.
    """
    return [[(code,sim)for code, sim in zip(codes, sims)]for codes, sims in zip(candidate_object.candidates.codes, candidate_object.candidates.similarity)]

def unflatten_list(lst, result=None):
    """Recursive function which, given a list of lists in its input, returns a single flat list.

    Args:
        lst (lst): _description_
        result (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if result is None:
        result = []
    for item in lst:
        if isinstance(item, list):
            unflatten_list(item, result)
        else:
            result.append(item)
    return result

def sort_lists(lista):
    """Function that list a list of tuples based on the second element of each tuple (the similarity).

    Args:
        lista (_type_): list of tuples. Each element is a tuple in the form (code, similarity_score)

    Returns:
        sorted list: _description_
    """
    return sorted(lista, key=lambda x: x[1], reverse=True)


def remove_duplicates(list_tuples, k_value, all_predictions = True):
    """Given a list of tuples, this function iterates over it eliminating those elements whose first 
    element (the code) is duplicated in the list, keeping the one with the highest value. 

    Args:
        list_tuples (list): List of tuples.  Each element is a tuple in the form (code, similarity_score)
        k_value (int): Max. number of elements to return. If all_predictions is True this value is not used. 
        all_predictions (bool, optional): If True, it returns a list without limiting its size. Defaults to True.

    Returns:
        list: _description_
    """
    # Create an empty dictionary
    max_similarities = {}
    # Iterate over thelist of tuples
    for s, sim in list_tuples:
        # If the string is not in the dictionary, add it with the current similarity vlaue
        if s not in max_similarities:
            max_similarities[s] = sim
        # If the string is in the dictionary, update the value if the current similarity is larger
        else:
            max_similarities[s] = max(max_similarities[s], sim)
        # Create a new list with the tuples from the dictionary
        result = [(s, sim) for s, sim in max_similarities.items()]
    
    return result if all_predictions else result[0:k_value]