import random
import pandas as pd

RAND_SEED = 23


def read_gazetteer_from_mongo(
        gazetteer_id: str,
        mongo_collection
) -> pd.DataFrame:
    cursor = mongo_collection.find({'gazetteer_id': gazetteer_id})
    df_gaz = pd.DataFrame(list(cursor))
    print("Gazetteer shape:", df_gaz.shape)
    if df_gaz.shape[0] == 0:
        raise Exception("Incorrectly specified gazetteer id: " + gazetteer_id)
    return df_gaz


def read_gazetteer_to_dict(
        df_gaz: pd.DataFrame,
        semantic_tag: bool = True,
        verbose: bool = True
):
    # Expected columns: code, term, semantic_tag, mainterm, language
    df_gaz["code"] = df_gaz["code"].astype(str)
    assert not df_gaz[["code", "term"]].duplicated().any()

    if semantic_tag:
        df_gaz["term"] = df_gaz.apply(
            lambda x: str(x.term) + " ["+str(x.semantic_tag)+"] ",
            axis=1
        )

    if verbose:
        print("Number of ambiguous terms:")
        print(pd.DataFrame({
            "abs": pd.Series(
                df_gaz["term"].value_counts() > 1
            ).value_counts(),
            "rel": pd.Series(
                df_gaz["term"].value_counts() > 1
            ).value_counts(normalize=True)
        }))
        print("\nNumber of codes with multiple synonyms:")
        # Terms
        print(pd.DataFrame({
            "abs": pd.Series(
                df_gaz["code"].value_counts() > 1
            ).value_counts(),
            "rel": pd.Series(
                df_gaz["code"].value_counts() > 1
            ).value_counts(normalize=True)
        }))

    # Create term-code dictionary
    dict_term_code = df_gaz.groupby(by=["term"], sort=True)["code"].apply(
        lambda x: sorted(set(x))
    ).to_dict()
    # eliminate ambiguity: we randomly select a code for the ambiguous terms
    arr_terms = sorted(dict_term_code.keys())
    for i, term in enumerate(arr_terms):
        random.seed(i * RAND_SEED)
        dict_term_code[term] = random.choice(dict_term_code[term])

    assert len(dict_term_code) == len(set(df_gaz["term"]))
    if verbose:
        print("\nNumber of terms:", len(dict_term_code))
        print("Number of codes:", len(set(dict_term_code.values())))

    return dict_term_code
