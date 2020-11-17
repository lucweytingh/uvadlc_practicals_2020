def print_all(*args):
    print(len(args))


class Test:
    def __init__(self):
        self.func = print_all

    def does(self, variable):
        self.func(variable)
