import unittest


class Example(unittest.TestCase):
    # method name must start with 'test'
    def test_upper(self):
        # convention is assert(testValue, case)
        self.assertEqual("upper".upper(), "UPPER")

    def testsomething_else(self):
        self.assertNotEqual("upper".upper(), "upper")
