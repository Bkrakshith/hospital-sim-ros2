from setuptools import setup

package_name = "g1_teleop_manager"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/keys.yaml", "config/zones.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rakshith",
    maintainer_email="bangrak2000@gmail.com",
    description="G1 mode FSM + command arbiter",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["manager_node = g1_teleop_manager.manager_node:main"]},
)
