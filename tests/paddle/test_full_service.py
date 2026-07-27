from __future__ import annotations

import unittest

from app.modules.v1.full.paddle import lines_to_text


class FullServiceTests(unittest.TestCase):
    def test_lines_to_text_joins_non_empty_lines(self) -> None:
        text = lines_to_text(
            [
                {"text": "Первая строка", "confidence": 0.99, "box": []},
                {"text": " ", "confidence": 0.20, "box": []},
                {"text": "Вторая строка", "confidence": 0.98, "box": []},
            ]
        )

        self.assertEqual(text, "Первая строка\nВторая строка")


if __name__ == "__main__":
    unittest.main()
