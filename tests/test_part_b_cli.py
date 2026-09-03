from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import run_puremagic_part_b_benchmarks as runner


class ResultFilenameTests(unittest.TestCase):
    def test_default_lambda_is_in_filename(self) -> None:
        with patch.object(sys, 'argv', ['runner']):
            args = runner.parse_args()
        self.assertEqual(args.results_file.name, 'puremagic_part_b_results_lambda_0.5.json')

    def test_selected_lambda_is_in_filename(self) -> None:
        for value in ('0.25', '0.75', '0.5000001'):
            with self.subTest(value=value):
                with patch.object(sys, 'argv', ['runner', '--magic-state-lambda', value]):
                    args = runner.parse_args()
                self.assertEqual(args.results_file.name, f'puremagic_part_b_results_lambda_{value}.json')

    def test_explicit_filename_is_preserved(self) -> None:
        with patch.object(sys, 'argv', [
            'runner', '--magic-state-lambda', '0.25', '--results-file', 'split.json',
        ]):
            args = runner.parse_args()
        self.assertEqual(args.results_file, Path('split.json'))


if __name__ == '__main__':
    unittest.main()
