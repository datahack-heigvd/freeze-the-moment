"""Cast association regressions. Run with just test."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def detection(ct, pid=-1, xy=(0.0, 0.0), mask=0):
    return dict(ct=ct, id=pid, xy=list(xy), mask=mask, team=ct, ref=ct == -1)


class CastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        bundle = Path(cls.tmp.name)
        scene = dict(
            n=16,
            hx=30,
            hy=15,
            corner=8,
            fps=25,
            fref=1000,
            cx=960,
            cy=540,
            roster=[dict(team=0, jersey="17", name="Player", goalie=False)],
        )
        (bundle / "scene.json").write_text(json.dumps(scene))
        np.savez(bundle / "scene.npz", cam=np.zeros((16, 14)))
        spec = importlib.util.spec_from_file_location("cast_test_subject", Path(__file__).parent.parent / "goal" / "cast.py")
        cls.cast = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, BUNDLE=str(bundle)):
            spec.loader.exec_module(cls.cast)

    @classmethod
    def tearDownClass(cls):
        cls.cast.Z.close()
        cls.tmp.cleanup()

    def test_empty_frames(self):
        self.assertEqual(self.cast.link([[], []]), [])
        seqs = self.cast.link([[detection(1)], [], [detection(1)]])
        self.assertEqual(len(seqs), 1)
        self.assertEqual([i for i, _ in seqs[0]], [0, 2])

    def test_link_rejects_opposite_team_and_referee(self):
        for ct in (0, -1):
            with self.subTest(colour=ct):
                seqs = self.cast.link([[detection(1)] for _ in range(3)] + [[detection(ct)]])
                self.assertEqual(sorted(map(len, seqs)), [1, 3])

    def test_unknown_colour_can_bridge_a_gap(self):
        seqs = self.cast.link([[detection(1)] for _ in range(3)] + [[detection(None)]])
        self.assertEqual(list(map(len, seqs)), [4])
        self.assertIsNone(self.cast.colour([(0, detection(1)), (1, detection(1))]))

    def test_join_respects_colour(self):
        first = [(i, detection(1)) for i in range(3)]
        for ct, count in ((0, 2), (-1, 2), (1, 1), (None, 1)):
            with self.subTest(colour=ct):
                second = [(i, detection(ct)) for i in range(4, 7)]
                self.assertEqual(len(self.cast.join([first, second])), count)

    def test_detection_rejects_wrong_feed_team_and_off_ice(self):
        inside = [[100, 100], [150, 100], [150, 220], [100, 220]]
        edge = [[100, 970], [150, 970], [150, 1079], [100, 1079]]
        scene = self.cast.S | dict(people=[[dict(mask=i, id=0, team=0) for i in range(3)]], outlines=[[inside, edge, inside]])
        with (
            patch.object(self.cast, "S", scene),
            patch.object(self.cast, "ray_to_ice", return_value=np.array([[0, 0], [0, 0], [40, 0]])),
            patch.object(self.cast, "frame_lab", return_value=np.zeros((1080, 1920, 3))),
            patch.object(self.cast, "jersey_px", return_value=np.ones((40, 3))),
            patch.object(self.cast, "team_of_track", return_value=1),
        ):
            dets = self.cast.detections()[0]
        self.assertEqual([d["mask"] for d in dets], [0, 1])
        self.assertEqual(dets[0]["id"], -1)
        self.assertEqual(dets[1]["id"], 0)
        self.assertIsNone(dets[1]["ct"])

    def test_foreground_boards_reject_staff_but_keep_clipped_player(self):
        cam = np.zeros(14)
        cam[11:14] = [0, -34, 15]
        xy = np.array([[0, -14], [0, -5], [-25, 0], [0, 14]])
        self.assertEqual(self.cast.feet_visible(cam, xy).tolist(), [False, True, True, True])
        self.assertEqual(self.cast.feet_visible(cam, np.empty((0, 2))).tolist(), [])
        cam[11:14] = [0, 0, 15]
        self.assertTrue(self.cast.feet_visible(cam, xy).all())

    def test_nearest_feed_fallback_respects_colour(self):
        for ct in (1, -1):
            with self.subTest(colour=ct):
                seq = [(i, detection(ct)) for i in range(16)]
                with (
                    patch.object(self.cast, "detections", return_value=[[detection(ct)] for _ in range(16)]),
                    patch.object(self.cast, "link", return_value=[seq]),
                    patch.object(self.cast, "sensor_paths", return_value={0: {i: np.zeros(2) for i in range(16)}}),
                    patch.object(self.cast, "jersey_px", return_value=np.ones((40, 3))),
                    patch.object(self.cast, "team_of_track", return_value=ct),
                    patch.object(
                        self.cast,
                        "S",
                        self.cast.S | dict(people=[[dict(mask=0)] for _ in range(16)], outlines=[[[[0, 0]]] for _ in range(16)]),
                    ),
                    patch("builtins.print"),
                ):
                    self.cast.main()
                tracks = json.loads(Path(self.cast.B, "cast.json").read_text())["tracks"]
                self.assertEqual(tracks[0]["mask"], [-1] * 16)
                self.assertEqual(tracks[1]["team"], ct)
                self.assertEqual(tracks[1]["mask"], [0] * 16)

    def test_frame_cache_decodes_once_and_is_bounded(self):
        self.cast.frame_lab.cache_clear()
        with patch.object(self.cast.cv2, "imread", return_value=np.zeros((8, 8, 3), np.uint8)) as read:
            self.cast.frame_lab(0)
            self.cast.frame_lab(0)
            self.cast.frame_lab(1)
        self.assertEqual(read.call_count, 2)
        self.assertEqual(self.cast.frame_lab.cache_info().currsize, 1)
        self.cast.frame_lab.cache_clear()


if __name__ == "__main__":
    unittest.main()
