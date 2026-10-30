import unittest
from data import split_data
from model import encode,normalize,TinyCNN
import torch
class ModelTests(unittest.TestCase):
    def test_family_split(self):
        splits=split_data()
        families=[{r['family'] for r in rows} for rows in splits]
        self.assertFalse(families[0]&families[1]);self.assertFalse(families[0]&families[2]);self.assertFalse(families[1]&families[2])
    def test_normalization_and_dimensions(self):
        self.assertEqual(normalize('پكيج ۱۲'),normalize('پکیج 12'))
        self.assertEqual(len(encode('سلام')),160)
        self.assertEqual(tuple(TinyCNN()(torch.tensor([encode('رادیاتور سرد است')])).shape),(1,8))
if __name__=='__main__':unittest.main()
