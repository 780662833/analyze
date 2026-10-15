"""Integration smoke test: uses FFmpeg to create a tiny artificial video and OpenCV fallback.
This checks wiring only; it is not a detector-accuracy benchmark.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class SyntheticPipelineTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
    def test_pipeline_outputs(self):
        with tempfile.TemporaryDirectory(prefix="omintel-") as tmp:
            work = Path(tmp)
            video = work / "synthetic.mp4"
            out = work / "result"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=1",
                "-f", "lavfi", "-i", "color=c=green:s=320x180:r=30:d=1",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=30:d=1",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
                "-map", "[v]", "-t", "3", str(video),
            ]
            subprocess.run(command, check=True, capture_output=True)
            run = ["python", str(ROOT / "run_analysis.py"), "--video", str(video), "--out", str(out),
                   "--scene-a", "opencv-fallback", "--scene-b", "ffmpeg-scdet", "--a-threshold", "20",
                   "--b-threshold", "10", "--min-gap", "0.3", "--match-tolerance", "0.1"]
            cp = subprocess.run(run, cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(cp.returncode, 0, msg=cp.stdout + "\n" + cp.stderr)
            for name in ["media_facts.json", "frames.csv", "shot_candidates.json", "shot_candidates.csv",
                         "provisional_shots.csv", "events.json", "evidence_index.json", "run_report.md", "analysis_run.json"]:
                self.assertTrue((out / name).exists(), msg=f"missing {name}")
            facts = json.loads((out / "media_facts.json").read_text(encoding="utf-8"))
            events = json.loads((out / "events.json").read_text(encoding="utf-8"))
            self.assertEqual(facts["frame_timeline"]["frame_count_decoded"], 90)
            self.assertEqual(facts["frame_timeline"]["missing_pts_count"], 0)
            self.assertGreaterEqual(events["event_count"], 1)
            self.assertTrue((out / "evidence").exists())
            self.assertTrue(all(e.get("evidence_status") == "complete" for e in events["events"]))
            for event in events["events"]:
                for frame in event["frames"].values():
                    self.assertEqual(len(frame["sha256"]), 64)
                    self.assertTrue((out / frame["path"]).is_file())

if __name__ == "__main__":
    unittest.main()
