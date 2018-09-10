import argparse


class Parser:

    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description=description)

    def add_arguments(self, arguments):
        for arg in arguments:
            self.parser.add_argument(*arg[0], **arg[1])

    def get_dictionary(self):
        return vars(self.parser.parse_args())