from setuptools import setup

package_name = "g1_mujoco_sim"
setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rakshith",
    maintainer_email="bangrak2000@gmail.com",
    description="MuJoCo simulation of the Unitree G1 speaking the unitree_hg low-level interface",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["sim_node = g1_mujoco_sim.sim_node:main"]},
)
