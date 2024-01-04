import torch

def add_eos():
    pass


def map_units(src_units, mapping_dict):
    mapped_units = []

    for unit in src_units.tolist():
        mapped_units.append(mapping_dict[str(unit)])

    return torch.LongTensor(mapped_units)