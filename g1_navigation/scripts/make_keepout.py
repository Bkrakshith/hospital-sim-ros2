#!/usr/bin/env python3
"""Write the Nav2 KeepoutFilter mask maps/stairs_keepout.{pgm,yaml} from the stair boxes of config/survey_route.yaml.

    python3 make_keepout.py --route config/survey_route.yaml --out maps/stairs_keepout

Every box of `stairs` (world frame: the two enclosed up-stairs and the two sealed down-stair pits of the main
building) is grown by `stairs_keepout_m` and written as keep-out (black). The mask keeps its fixed frame:
resolution 0.1 m, origin (-52, -27), 870 x 670 cells, so it stays aligned with every map of the same world frame.
"""
import argparse
import os

import numpy as np
import yaml

RES, X0, Y0, W, H = 0.1, -52.0, -27.0, 870, 670


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    route = yaml.safe_load(open(a.route))
    grow = float(route.get("stairs_keepout_m", 0.25))
    xs = X0 + (np.arange(W) + 0.5) * RES
    ys = Y0 + (np.arange(H) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    keep = np.zeros((H, W), bool)
    names = []
    for s in route.get("stairs", []):
        x0, y0, x1, y1 = s["box"]
        keep |= (X >= x0 - grow) & (X <= x1 + grow) & (Y >= y0 - grow) & (Y <= y1 + grow)
        names.append(s["name"].split(" (")[0])
    img = np.where(keep, 0, 254).astype(np.uint8)[::-1]
    with open(a.out + ".pgm", "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (W, H))
        f.write(img.tobytes())
    with open(a.out + ".yaml", "w") as f:
        f.write("# Nav2 KeepoutFilter mask (NOT a map), written by scripts/make_keepout.py from config/survey_route.yaml:\n"
                f"# {'; '.join(names)}, each grown by {grow} m.\n"
                "# The up-stairs are enclosed (screens + FD30S doors) and the down-stairs are sealed by fixed screens:\n"
                "# the mask only stops the planner from entering them. World / map frame. Used by navigation.launch.py\n"
                "# keepout:=true (costmap_filter_info_server + filter mask server).\n")
        f.write(f"image: {os.path.basename(a.out)}.pgm\nmode: scale\nresolution: {RES}\n"
                f"origin: [{X0}, {Y0}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    print(f"wrote {a.out}.yaml ({W}x{H} @ {RES} m), {len(names)} boxes, {int(keep.sum())} keep-out cells")


if __name__ == "__main__":
    main()
