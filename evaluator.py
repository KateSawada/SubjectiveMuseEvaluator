from typing import List, Callable

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from .units import EvaluateUnit


class Evaluator:
    def __init__(
            self,
            methods: List[EvaluateUnit],
            filenames: List[str],
            loader: Callable[[str], np.ndarray],
    ):
        self.methods = methods
        self.filenames = filenames
        self.loader = loader

    def run(self, output_file: str):
        """run evaluations

        Args:
            output_file (str): output file name
        """
        progress_bar = tqdm(range(len(self.filenames)))
        progress_bar.set_description("samples")

        for idx_samples in progress_bar:
            # define function ran by map
            def _run_method(method):
                sample = self.loader(self.filenames[idx_samples])
                method(idx_samples, sample)
            # execute method
            for idx_methods in range(len(self.methods)):
                tuple(map(_run_method, self.methods))

        # merge all results
        results = {}
        for idx_methods in range(len(self.methods)):
            results = dict(results, **self.methods[idx_methods].to_dict())
        results = OmegaConf.create(results)
        print(results)
        OmegaConf.save(results, output_file)
