import os
import pandas as pd
import os, sys
import random
from tqdm import tqdm

general_path = os.getcwd().split("deepspanorm")[0]+"deepspanorm/"
sys.path.append(general_path+'src/')



# Most of this code is taken from DILBERT repository. It should be changed, recoded or referenced properly.
#REMOVE
def is_equal(label_1: str, label_2: str) -> bool:
    """
    Comparing composite concept_ids
    """
    return len(set(label_1.replace('+', '|').split("|")).intersection(set(label_2.replace('+', '|').split("|")))) > 0
    

def find_last_occurence(ordered_labels, label) -> int:
    for i in range(len(ordered_labels) - 1, 1, -1):
        if is_equal(ordered_labels[i][0], label): return i
    return 0


## Files to generate random files. The previous code can be erased (i think)

def is_ambiguous(term, lista):
    cuenta = lista.count(term)
    return True if cuenta>1 else False

def build_triplets_to_output(lista_preselected, num_positive, num_negative):
    """
    Construye una lista de tripletas con el número de muestras positivas y negativas
    pasadas como atributo. La selección se hace aleatoria a partir de la selección
    de datos realizada previamente (y contenida en la lista_preselected)
    """
    output_list = list()
    for mention in tqdm(lista_preselected, total=len(lista_preselected)):
        # Separate tuple
        anchor, positive_elements, negative_elements = mention

        # Generate n_positive elements from positive_elements. If there are not enough
        # Select n_positive elements from positive_elements, with replacement
        # If the list is not long enough, add more elements with replacement
        selected_positives = list()
        if num_positive <= len(positive_elements):
            # If the list is long enough, select n_positive elements without replacement
            seleccion = random.sample(positive_elements, k=num_positive)
            for selected in seleccion:
                selected_positives.append(selected)
        elif len(positive_elements) == 0:
            seleccion = [anchor]*num_positive
            for selected in seleccion:
                selected_positives.append(selected)
        else:
            # If the list is not long enough, add more elements with replacement
            for _ in range(num_positive):
                # Select elements from positive_elements, with replacement
                selected_positives.append(random.choice(positive_elements))
            
        # Generate the positive pairs.
        positive_pairs = [(anchor, positive) for positive in selected_positives]
        # Incorpore also the negative pairs
        repeated_pairs = [(pair[0], pair[1]) for pair in positive_pairs for _ in range(num_negative)]
        positive_negative_pairs =  [(anchor, positive, random.sample(negative_elements, k=1)[0]) for anchor, positive in repeated_pairs]
        output_list.extend(positive_negative_pairs)
    return output_list

def get_parent_codes_dict(lista_codigos_corpus, vocab):
    """
    Esta función genera un diccionario en el que a cada código presente en el corpus
    se le extraen del diccionario los códigos padres que deben tenerse en cuenta para 
    buscar sinónimos, así como sus strings.
    """
    # Esto sirve para generar un diccionario de strings y codigo spositivos
    dict_positives_codes = dict()
    dict_positives_str = dict()
    
    codigos_not_found = 0
    for codigo in lista_codigos_corpus:
        try:
            # Para cada código del corpus, vamos a coger los códigos padres mirándolos en el vocabulario
            parent_codes = eval(vocab[vocab.code==codigo].parents.iloc[0])
            # Cogemos las strings únicas asociadas a cada uno de esos códigos padres y al código actual 
            parent_strings = list(vocab[vocab.code.isin(parent_codes)].term.unique()) + list(vocab[vocab.code==codigo].term.unique())
            # Posteriormente generaremos un nuevo diccionario en el que se mapeen los códigos a sus respectivas strings
            dict_positives_codes[codigo] = {
                                            "strings":parent_strings,
                                            "codes": parent_codes
                                            }
        except:
            codigos_not_found +=1

    return dict_positives_codes

def generate_candidate_list_parents(corpus, vocab, dict_positive_codes, num_positive, num_negative):
    """
    Esta función genera una lista que devuelve una lista de longitud del corpus (un elemento por mención). 
    En el que cada elemento es un triple que contiene: 
        - Anchor o mención
        - Strings candidatas a ser positivos
        - Strings candidatas a ser negativas (se hace una selección random).
    """
    lista_salida_pos = list()
    # Aquí poner unique en los codigos de iteración para 
    for span in corpus.span.to_list():
        #Get code asociado al span.
        codigo_asociado_a_span = corpus[corpus.span==span].code.iloc[0]
        # Buscamos elementos negativos que no están en la parte de códigos:
        try:
            codigos_to_select_in_negatives = vocab[~vocab.code.isin(dict_positives_codes[codigo_asociado_a_span]["codes"])].code.to_list()
            strings_to_select = random.choices(vocab[vocab.code.isin(codigos_to_select_in_negatives)].term.to_list(),
                                              k=(num_positive*3)*num_negative)
             # Guardamos todo en una lista:
            lista_salida_pos.append((span,
                                     dict_positives_codes[codigo_asociado_a_span]["strings"],
                                     random.choices(strings_to_select,k=(num_positive*10)*num_negative)))
        
        except: 
            # Si hay excepción es porque por algún motivo hay un código del training que ya es obsoleto
            # con el diccionario que se está usando, pero podemos seguir generando triplets.
            strings_to_select = vocab.term.to_list()
             # Guardamos todo en una lista:
            lista_salida_pos.append((span,
                                     vocab[vocab.code==codigo_asociado_a_span].term.to_list(),
                                     random.choices(strings_to_select,k=(num_positive*10)*num_negative)))
 
    return lista_salida_pos

def save_triplets_to_file(filename, lista_triplets):
    with open(filename, 'w', encoding='utf-8') as output_stream: 
        # Iteramos por cada entidades y etiquetas de train y las etiquetas del diccionario que se combinarán.
        for anchor, positive, negative in lista_triplets:
            output_stream.write(f'{anchor}\t{positive}\t{negative}\n')
            
def generate_triplets(path_to_vocab, synonyms, ignore_ambiguous, from_corpus, path_to_corpus, ignore_composite,
                     num_positive, num_negative, strategy,output_file,include_semantic_tag,
                      composite_separator = "+", non_code_str = "NOMAP",unique_spans=True ):
    # Read vocab
    vocab = pd.read_csv(path_to_vocab,sep="\t")
    # If generate data from corpus
    if from_corpus:
        ## Read corpus
        corpus = pd.read_csv(path_to_corpus, sep="\t") 
        corpus["code"] = corpus.code.apply(lambda x:[int(x) if x != non_code_str else x for x in x.split(composite_separator)])
        ## Prepare corpus
        if ignore_composite:
            corpus = corpus[corpus.semantic_rel!="COMPOSITE"].reset_index(drop=True)
        
        corpus = corpus.explode("code")
        corpus["code"] = corpus.code.astype('str')
        corpus = corpus.drop_duplicates()
        
        ## Prepare vocab
        # Include semantic tag in string to try to improve performance
        if include_semantic_tag:
            vocab["term"] = vocab.apply(lambda x: str(x.term) + " ["+str(x.semantic_tag)+"] ", axis=1)
        # Ambiguous behaviour
        if ignore_ambiguous:
            lista_strings = vocab.term.to_list()
            vocab["is_ambiguous"] = vocab.term.apply(lambda x: is_ambiguous(x, lista_strings))
            # Delete ambihuous
            vocab = vocab[vocab.is_ambiguous==False].reset_index(drop=True)
        # Prepare synoynms
        if synonyms=="comma":  # NEED TO FIX
            vocab = vocab.groupby(['code'])['term'].apply(lambda x: ','.join(x)).reset_index()
            vocab["code"] = vocab.code.astype('str')
        elif synonyms == "individual":
            vocab["code"] = vocab.code.astype('str')
    else:
        print("Todavía no se ha desarrollado la función para generar triplets a partir de sólo un vocabulario")
    if unique_spans:
        corpus = corpus.drop_duplicates(subset=['span'])
    # Para generar siguiendo en corpus habrá dos opciones parent y random
    if strategy == "parents":
        # En esta estrategia, primero extraemos del vocabulario los códigos padres
        # de cada concepto del corpus que se considerarán para generar sinónimo
        dict_positives_codes = get_parent_codes_dict(corpus.code.to_list(), vocab)
        # Posteriormente, generamos una lista de tuplas en el que cada elemento 
        # contendrá la mención o "anchor", strings positivas candidatas, y un subset de 
        # strings negativas seleccionadas aleatoriamente (siempre habrá suficientes para
        # después). 
        lista_salida_pos = generate_candidate_list_parents(corpus, vocab,dict_positives_codes, num_positive, num_negative)
        # Despues generamos las tripletas
        positive_negative_pairs = build_triplets_to_output(lista_salida_pos, num_positive, num_negative)
        # Por ultimo guardamos el archivo
        save_triplets_to_file(output_file, positive_negative_pairs)
    elif strategy == "random":
        # Generamos una lista de tuplas en el que cada elemento 
        # contendrá la mención o "anchor", strings positivas que en este caso serán sinónimos, y un subset de 
        # strings negativas seleccionadas aleatoriamente
        # En este caso el diccionario de códigos tenidos en cuenta para generar positivos será un diccionario 
        # vacío (no se tiene en cuenta la estructura jerárquica de la temrinología).
        dict_positives_codes = {}
        lista_salida_pos = generate_candidate_list_parents(corpus, vocab,dict_positives_codes, num_positive, num_negative)
        # Despues generamos las tripletas
        positive_negative_pairs = build_triplets_to_output(lista_salida_pos, num_positive, num_negative)
        # Por ultimo guardamos el archivo
        save_triplets_to_file(output_file, positive_negative_pairs)
    