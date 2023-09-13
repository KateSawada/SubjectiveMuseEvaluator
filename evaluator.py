from typing import List

from .units import EvaluateUnit


class Evaluator:
    def __init__(self, methods: List[EvaluateUnit]):
        self.methods = methods

    def run(self, output_file: str):
        """run evaluations

        Args:
            output_file (str): output file name
        """
        # TODO: implement
