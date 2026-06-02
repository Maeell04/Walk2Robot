try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    rclpy = None
    Node = object
    String = None


class NoeudInferencePhaseMarche(Node):
    """Recoit des caracteristiques et publie une phase de marche simulee."""

    def __init__(self):
        super().__init__("noeudinferencephasemarche")
        self.compteur = 0
        if rclpy is not None:
            self.pub = self.create_publisher(String, "/phase_marche", 10)
            self.timer = self.create_timer(0.1, self.boucle)

    def boucle(self):
        self.compteur += 1
        message = String()
        message.data = "phase_marche: appui_droit"
        self.pub.publish(message)
        self.get_logger().info("Message publie en francais")


def main(args=None):
    if rclpy is None:
        print("Recoit des caracteristiques et publie une phase de marche simulee.")
        return
    rclpy.init(args=args)
    noeud = NoeudInferencePhaseMarche()
    rclpy.spin(noeud)
    noeud.destroy_node()
    rclpy.shutdown()
