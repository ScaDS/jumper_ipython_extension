#!/usr/bin/env python3
"""Fetch the model files and sample images used by ai_review_advanced.ipynb.

Run from anywhere:

    python demos/fetch_resources.py

demos/resources/ is intentionally kept out of git (see root .gitignore); this
script re-materializes its contents.

Notes on weights:
  * OpenPose COCO caffemodel is downloaded from a public HuggingFace mirror
    (the original CMU host is frequently offline).
  * The genuine ArcFace IR-50 MS1M weights (backbone_ir50_ms1m_epoch120.pth)
    are only distributed via a gated Google Drive by face.evoLVe. This script
    instead materializes an architecture-compatible weights file from
    arcface_backbone.Backbone so the notebook's load_state_dict + forward pass
    run end to end. Drop the real .pth in place to use trained weights - the
    architecture is identical, so it loads without changes.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "resources"
HERE.mkdir(exist_ok=True)

DOWNLOADS = {
    "pose_deploy_linevec.prototxt": (
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
        "openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
    ),
    "pose_iter_440000.caffemodel": (
        "https://huggingface.co/camenduru/openpose/resolve/main/"
        "models/pose/coco/pose_iter_440000.caffemodel"
    ),
    "group.jpg": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/"
        "OpenPose-Multi-Person/group.jpg"
    ),
    "boy.png": (
        "https://raw.githubusercontent.com/MarcoForte/FBA_Matting/master/"
        "examples/images/troll.png"
    ),
    "arcface_backbone.py": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/"
        "Face-Recognition-with-ArcFace/backbone.py"
    ),
}


def curl(url: str, dest: Path):
    print(f"  -> {dest.name}")
    subprocess.run(
        [
            "curl",
            "-sL",
            "--fail",
            "--max-time",
            "600",
            "-o",
            str(dest),
            url,
        ],
        check=True,
    )


def make_arcface_weights():
    """Materialize an architecture-compatible IR-50 weights file + face folder."""
    sys.path.insert(0, str(HERE))
    import cv2
    import torch

    from arcface_backbone import Backbone

    weights = HERE / "backbone_ir50_ms1m_epoch120.pth"
    if not weights.exists():
        print("  -> backbone_ir50_ms1m_epoch120.pth (generated placeholder)")
        model = Backbone([112, 112])
        model.eval()
        torch.save(model.state_dict(), weights)

    faces = HERE / "arcface_faces" / "person_a"
    faces.mkdir(parents=True, exist_ok=True)
    if not any(faces.iterdir()):
        print("  -> arcface_faces/person_a/*.jpg")
        src = cv2.imread(str(HERE / "group.jpg"))
        for i in range(3):
            cv2.imwrite(str(faces / f"face_{i}.jpg"), cv2.resize(src, (160, 160)))


def main():
    print("Fetching notebook resources into", HERE)
    for name, url in DOWNLOADS.items():
        dest = HERE / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  == {name} (exists, skipping)")
            continue
        curl(url, dest)
    make_arcface_weights()
    print("Done.")


if __name__ == "__main__":
    main()
