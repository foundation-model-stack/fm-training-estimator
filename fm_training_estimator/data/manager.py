# Local
from ..utils import extract_model_features


class Format:
    """A class to track the various data formats used for lookup/regressor.

    Stores the features used/predicted as strings.
    """

    def __init__(self, name, X, Y):
        self.name = name
        self.X = X
        self.Y = Y

    def get_all_columns_string(self):
        return self.X + "," + self.Y

    def get_empty_key_dict(self):
        res = {}
        for x in self.X.split(","):
            res[x] = None

        return res


"""
This is the list of accepted/known data formats.

Only one of the following is a valid format for csv files for lookup and for any trained regression models.

When new formats are to be supported this list is to be updated with a new Format object.
"""
formats = [
    Format(
        "v1",
        "model_name,number_gpus,batch_size,seq_len",
        "tokens_per_second,memory,memory_act",
    ),
    Format(
        "v2",
        "model_arch,model_hidden_size,model_intermediate_size,model_num_attn_heads,model_num_hidden_layers,model_num_key_value_heads,number_gpus,batch_size,seq_len",
        "tokens_per_second,memory,memory_act",
    ),
    Format(
        "v3",
        "model_arch,model_hidden_size,model_intermediate_size,model_num_attn_heads,model_num_hidden_layers,model_num_key_value_heads,method,gpu_model,number_gpus,batch_size,seq_len",
        "tokens_per_second,memory,memory_act",
    ),
]


def lookup_format_version(data_keys):
    """
    Given a string of comma separated keys, looks up any matching defined format version.

    The input included both X and Y columns in that order, like the header of
    the CSV used to train/lookup.
    """
    for f in formats:
        if data_keys == f.get_all_columns_string():
            return f.name

    return "undefined"


def get_format_by_version(version):
    """Given a version string, return the relevant Format object."""
    for f in formats:
        if f.name == version:
            return f

    return None


def format_query(partials, version, only_values=False):
    """
    Format a query for a given version using the provided partial information.

    If only_values is False, returns a dictionary of key-values according to the format.
    If it is true, returns the values as an array. The former is needed for direct
    lookup in the lookup module, while the latter is used by the regressor.
    """

    vf = get_format_by_version(version)

    # TODO: vf can be None here, if an unsupported format is seen.

    query = vf.get_empty_key_dict()

    # fill in all matching fields from the input, if present in desired version
    for k, v in partials.items():
        if k in query:
            query[k] = v

    # Handle changes for other model versions here

    if version == "v2" or version == "v3":
        model_features = extract_model_features(partials["model_name"])
        for k, v in model_features.items():
            if k in query:
                query[k] = v

    # TODO: validate that all fields are filled in here, no None's present
    # print(query)

    if not only_values:
        return query
    else:
        return query.values()
