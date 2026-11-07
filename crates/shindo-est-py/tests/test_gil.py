# Copyright 2026 oruponu
# SPDX-License-Identifier: MIT
"""Checks the GIL helper against calls known to hold or release the GIL."""

import time
import unittest

from tests import gil


@unittest.skipUnless(gil.is_enabled(), "the GIL is disabled")
class IsReleasedDuringTest(unittest.TestCase):

    def test_false_for_a_call_that_holds_the_gil(self):
        # sum runs in C over the range without releasing the GIL.
        self.assertFalse(gil.is_released_during(lambda: sum(range(10**6))))

    def test_true_for_a_call_that_releases_the_gil(self):
        self.assertTrue(gil.is_released_during(lambda: time.sleep(0.01)))


if __name__ == "__main__":
    unittest.main()
