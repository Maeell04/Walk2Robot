from setuptools import setup


package_name = "phase_marche_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Equipe projet",
    maintainer_email="etudiant@example.com",
    description="Noeuds ROS2 pour la phase de marche.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "publieur_caracteristiques_marche = phase_marche_ros2.publieur_caracteristiques_marche:main",
            "noeud_inference_phase_marche = phase_marche_ros2.noeud_inference_phase_marche:main",
            "controleur_assistance_robot = phase_marche_ros2.controleur_assistance_robot:main",
            "moniteur_performance_ros2 = phase_marche_ros2.moniteur_performance_ros2:main",
        ],
    },
)
