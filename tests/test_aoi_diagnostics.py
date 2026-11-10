import unittest
import numpy as np
from tools.diagnose_aois import summary


class DiagnosticMetricTests(unittest.TestCase):
    def test_threshold_retention_components_and_absent_class(self):
        labels=np.array([[1,1,0],[0,0,1]],dtype="uint8")
        confidence=np.array([[.3,.5,.9],[.9,.9,.7]])
        low=summary(labels,confidence,{0:"background",1:"eye",2:"missing"},.4)
        high=summary(labels,confidence,{0:"background",1:"eye",2:"missing"},.6)
        self.assertEqual(low[1]["raw_pixels"],3)
        self.assertEqual(low[1]["retained_pixels"],2)
        self.assertEqual(high[1]["retained_pixels"],1)
        self.assertEqual(high[1]["components"],1)
        self.assertEqual(high[2]["retained_fraction"],"")
        self.assertEqual(high[2]["mean_confidence"],"")
        self.assertEqual(high[2]["components"],0)
