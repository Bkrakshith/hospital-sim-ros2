from setuptools import setup

package_name = "g1_teleop_keyboard"
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
    description="Terminal keyboard teleop for the G1 -> /g1/teleop/key",
    license="Apache-2.0",
    entry_points={"console_scripts": ["keyboard_node = g1_teleop_keyboard.keyboard_node:main"]},
)
