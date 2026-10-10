import os
from glob import glob

from setuptools import setup

package_name = "g1_mission"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, package_name + ".skills"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "scripts"), glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rakshith",
    maintainer_email="bangrak2000@gmail.com",
    description="Scrub-nurse mission executive and skill library for the G1",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["mission_node = g1_mission.mission_node:main"]},
)
