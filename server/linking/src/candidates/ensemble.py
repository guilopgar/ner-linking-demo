import os, sys
general_path = os.getcwd().split("deepspanorm")[0]+"deepspanorm/"
sys.path.append(general_path+'src/')
from utils.data_preparation import check_path_ending
from utils.data_structures import predictions
import copy
from collections import OrderedDict


class EnsembleCandidateGeneration:
    def __init__(
            self, list_ensembles,
            remove_duplicates=True, k=None
    ) -> None:
        """
        :param list_ensembles
            List of candidates from different models
        """
        # Build detokenizer
        self.remove_duplicates = remove_duplicates
        self.k = k
        self.join_candidates(list_ensembles)


    def join_candidates(self, list_ensembles):
        # Hacemos deepcopy to avoid copying with references
        list_ensembles = copy.deepcopy(list_ensembles)

        # Initialize result object with sublists from the first object in the list
        resultado = predictions(
            list_ensembles[0].codes,
            list_ensembles[0].similarity,
            list_ensembles[0].texts
        )

        # Add elements from the objects from 1 to the end of the list of candidates.
        for candidate_model in list_ensembles[1:]:
            for i in range(len(candidate_model.codes)):
                resultado.codes[i].extend(candidate_model.codes[i])
                resultado.similarity[i].extend(candidate_model.similarity[i])
                resultado.texts[i].extend(candidate_model.texts[i])

        # Sort the results
        out_codes, out_sim, out_texts = list(), list(), list()
        for ind in range(0, len(resultado.codes)):
            # Iterating over query mentions
            sorted_codes, sorted_sim, sorted_texts = (
                list(t) for t in zip(*sorted(
                    zip(
                        resultado.codes[ind],
                        resultado.similarity[ind],
                        resultado.texts[ind]
                    ),
                    key=lambda x: x[1], reverse=True
                ))
            )
            out_codes.append(sorted_codes)
            out_sim.append(sorted_sim)
            out_texts.append(sorted_texts)

        # If remove_duplicates is True
        if self.remove_duplicates:
            # Delete second elements
            rem_dupl = list()
            for code, sim, text in zip(out_codes, out_sim, out_texts):
                # Iterating over query mentions
                set_temporal = set()
                lista_inner = list()
                for a, b, c in list(zip(code, sim, text)):
                    # Si el descriptor no está en el set
                    if c not in set_temporal:
                        # Añadimos ese elemento y lo guardamos en la lista
                        set_temporal.add(c)
                        lista_inner.append((a, b, c))
                rem_dupl.append(lista_inner)
            # Divide again in three lists
            out_codes = [[x[0] for x in sublist] for sublist in rem_dupl]
            out_sim = [[x[1] for x in sublist] for sublist in rem_dupl]
            out_texts = [[x[2] for x in sublist] for sublist in rem_dupl]

        # For each query, calculate the number of texts that produce
        # self.k distinct codes, and produce the codes_candidates
        # List of OrderedDict
        self.codes_candidates = []
        arr_max_i = []
        for arr_codes, arr_texts in zip(out_codes, out_texts):
            assert len(arr_codes) == len(arr_texts)
            # Iterate over queries
            dict_code_texts = OrderedDict()
            i = 0
            n_codes = 0
            while n_codes < self.k:
                # i < len(arr_codes[i]),
                # since max_n_texts were retrieved
                text = arr_texts[i]
                code = arr_codes[i]
                if code in dict_code_texts:
                    dict_code_texts[code].append(text)
                else:
                    dict_code_texts[code] = [text]
                    n_codes += 1
                i += 1

            arr_max_i.append(i)
            # Sort the texts associated to each code
            for code in dict_code_texts:
                texts = dict_code_texts[code]
                # Duplicated texts associated with the same code
                # are not expected
                assert len(texts) == len(set(texts))
                dict_code_texts[code] = sorted(texts)

            self.codes_candidates.append(dict_code_texts)

        for j in range(len(arr_max_i)):
            # Iterate over queries
            out_codes[j] = out_codes[j][:arr_max_i[j]]
            out_sim[j] = out_sim[j][:arr_max_i[j]]
            out_texts[j] = out_texts[j][:arr_max_i[j]]

        # Create the output object
        # Prepare data to be stored in predictions variable:
        self.candidates = predictions(out_codes, out_sim, out_texts)
        print("Joined!")
