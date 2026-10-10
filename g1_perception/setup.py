from glob import glob

from setuptools import setup

package_name = "g1_perception"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rakshith",
    maintainer_email="bangrak2000@gmail.com",
    description="Instrument OBB detection + grasp pose, surgeon hand detection (g1 scrub-nurse mission)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "detector_node = g1_perception.detector_node:main",
        "surgeon_hand_node = g1_perception.surgeon_hand_node:main",
    ]},
)
