import glob
import os

from setuptools import setup

package_name = "g1_controller"
data = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
for d in sorted(glob.glob("policies/*")):
    if os.path.isdir(d):
        data.append((f"share/{package_name}/{d}", [f for f in glob.glob(f"{d}/*") if os.path.isfile(f)]))
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=data,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rakshith",
    maintainer_email="bangrak2000@gmail.com",
    description="G1 policy runner (ONNX + JSON sidecar) -> /lowcmd",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["controller_node = g1_controller.controller_node:main"]},
)
