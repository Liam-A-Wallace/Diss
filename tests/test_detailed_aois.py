"""Integrity checks for detailed masks; these do not measure anatomical accuracy."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from analysis.detailed_aois import build_regions, LABELS


class DetailedIntegrityTests(unittest.TestCase):
    def test_manifest_save_recovers_from_transient_windows_file_lock(self):
        from analysis.face_aois import save_manifest
        original = Path.replace
        attempts = []
        def briefly_locked(source, target):
            attempts.append(target)
            if len(attempts) == 1:
                raise PermissionError("temporary reader lock")
            return original(source, target)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"manifest.json"
            path.write_text('{"old": true}')
            with patch.object(Path, "replace", autospec=True, side_effect=briefly_locked), patch("time.sleep"):
                save_manifest(path, {"new": True})
            self.assertEqual(json.loads(path.read_text()), {"new": True})
            self.assertEqual(len(attempts), 2)

    def test_native_lips_nose_and_accessories_survive_without_mesh(self):
        raw = np.array([[2,10,11,12,15,4,6,8]], dtype=np.uint8)
        names = {2:"nose",10:"mouth",11:"u_lip",12:"l_lip",15:"ear_r",4:"l_eye",6:"l_brow",8:"l_ear"}
        mask, flags = build_regions(raw, np.ones(raw.shape), names)
        np.testing.assert_array_equal(mask, [[2,10,11,12,15,255,255,255]])
        self.assertTrue(any("landmarks_unavailable" in flag for flag in flags))

    def test_no_geometry_can_relabel_uncertain_or_core_feature_pixels(self):
        # Exercise all region builders on a synthetic upright face, including
        # a confident mouth patch and low-confidence skin across the lower face.
        points = np.tile([50.,50.,0.], (478,1))
        coordinates = {33:(25,35),133:(40,35),362:(60,35),263:(75,35),
            152:(50,95),10:(50,5),1:(50,50),234:(10,50),454:(90,50),
            105:(30,25),334:(70,25),50:(20,55),187:(25,65),
            280:(80,55),411:(75,65),17:(50,75),61:(35,70),291:(65,70)}
        for k,value in coordinates.items(): points[k,:2]=value
        raw = np.ones((100,100), dtype=np.uint8)
        raw[33:38,25:40]=4; raw[33:38,60:75]=5
        raw[23:27,25:40]=6; raw[23:27,60:75]=7
        raw[68:74,35:65]=11
        confidence = np.ones(raw.shape); confidence[80:]=.2
        mask, _ = build_regions(raw, confidence,
            {1:"skin",4:"l_eye",5:"r_eye",6:"l_brow",7:"r_brow",11:"u_lip"}, points)
        self.assertTrue(np.all(mask[80:]==255))
        self.assertTrue(np.all(mask[68:74,35:65]==11))
        self.assertTrue(set(np.unique(mask)) <= set(LABELS))
        derived = (mask>=19)&(mask<=24)
        self.assertTrue(np.all(raw[derived]==1))

    def test_invalid_mesh_is_withheld(self):
        points=np.full((478,3), np.nan)
        mask, flags=build_regions(np.array([[4]]), np.ones((1,1)), {4:"l_eye"}, points)
        self.assertEqual(int(mask[0,0]),255)
        self.assertTrue(flags)

    def test_forehead_reaches_parser_hairline_above_mesh_oval(self):
        from analysis.detailed_aois import OVAL
        points = np.tile([50.,50.,0.],(478,1))
        for k,angle in zip(OVAL,np.linspace(-np.pi/2,3*np.pi/2,len(OVAL),endpoint=False)):
            points[k,:2] = [50+40*np.cos(angle),55+35*np.sin(angle)]
        for k,q in {33:(25,40),133:(40,40),362:(60,40),263:(75,40),
                    152:(50,90),1:(50,55),61:(35,70),291:(65,70),17:(50,75)}.items():
            points[k,:2]=q
        for k in [127,70,63,105,66,107,9,336,296,334,293,300,356]:
            points[k,1]=32
        raw=np.zeros((100,100),dtype=np.uint8)
        raw[5:91,10:91]=1
        raw[0:5,10:91]=13
        mask,_=build_regions(raw,np.ones(raw.shape),{0:"background",1:"skin",13:"hair"},points)
        self.assertEqual(int(mask[7,50]),19)
        self.assertEqual(int(mask[2,50]),13)
