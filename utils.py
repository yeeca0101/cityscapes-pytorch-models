


def get_total_params(model):
    return sum([p.numel() for p in model.parameters()])

def print_count_params(model):
    print('{:,}'.format(get_total_params(model)))