class DotDict(dict):
    def __init__(self, d: dict={}):
        super().__init__()
        for key, value in d.items():
            self[key] = DotDict(value) if type(value) is dict else value

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__