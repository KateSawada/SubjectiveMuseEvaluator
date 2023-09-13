from typing import List

import numpy as np

from .units import EvaluateUnit


class Evaluator:
    def __init__(
            self,
            methods: List[EvaluateUnit],
            samples: List[np.ndarray],
    ):
        self.methods = methods
        self.samples = samples

    def run(self, output_file: str):
        """run evaluations

        Args:
            output_file (str): output file name
        """
        # TODO: implement
