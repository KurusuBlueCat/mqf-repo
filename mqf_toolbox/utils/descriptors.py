class MustSet:
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name

    def __init__(self, default=None):
        self.value = default

    def __get__(self, obj, owner):
        value = getattr(obj, self.private_name)
        if value is None:
            raise AttributeError(self.public_name + ' was not set!')
        return value

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)
