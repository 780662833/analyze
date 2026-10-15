import unittest
from omintel.fusion import fuse_candidates, nearest_frame_index

class FusionTests(unittest.TestCase):
    def test_one_to_one_matching(self):
        a = [
            {"boundary_id": "A1", "time_s": 1.00, "detector": "A", "score": 10},
            {"boundary_id": "A2", "time_s": 2.00, "detector": "A", "score": 20},
        ]
        b = [
            {"boundary_id": "B1", "time_s": 1.04, "detector": "B", "score": 9},
            {"boundary_id": "B2", "time_s": 3.00, "detector": "B", "score": 11},
        ]
        fused = fuse_candidates(a, b, 0.1)
        self.assertEqual(len(fused), 3)
        self.assertEqual(sum(x["status"] == "agreement" for x in fused), 1)
        self.assertEqual(sum(x["status"] == "scene_a_only" for x in fused), 1)
        self.assertEqual(sum(x["status"] == "scene_b_only" for x in fused), 1)

    def test_matching_maximizes_pair_count_before_delta(self):
        # A closest-first greedy matcher would pair A2-B1 and lose a second valid pair.
        a = [
            {"boundary_id": "A1", "time_s": 0.00, "detector": "A", "score": 1},
            {"boundary_id": "A2", "time_s": 0.03, "detector": "A", "score": 1},
        ]
        b = [
            {"boundary_id": "B1", "time_s": 0.02, "detector": "B", "score": 1},
            {"boundary_id": "B2", "time_s": 0.05, "detector": "B", "score": 1},
        ]
        fused = fuse_candidates(a, b, 0.03)
        self.assertEqual(len(fused), 2)
        self.assertEqual(sum(x["status"] == "agreement" for x in fused), 2)

    def test_pts_nearest_frame(self):
        rows = [{"frame_no": 0, "pts_time_s": 0.0}, {"frame_no": 1, "pts_time_s": 0.04},
                {"frame_no": 2, "pts_time_s": 0.08}]
        self.assertEqual(nearest_frame_index(0.06, rows), 1)
        self.assertEqual(nearest_frame_index(0.079, rows), 2)

if __name__ == "__main__":
    unittest.main()
