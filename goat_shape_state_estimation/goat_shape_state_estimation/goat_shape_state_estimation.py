import torch
import roma
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Float32


# Constants for input indices
INDEX_IMU_ANGULAR_VELOCITY = 3
INDEX_LINEAR_ACCELERATION = INDEX_IMU_ANGULAR_VELOCITY + 3
INDEX_MEASURED_WHEEL_VELOCITY = INDEX_LINEAR_ACCELERATION + 3
INDEX_COMMANDED_VELOCITY = INDEX_MEASURED_WHEEL_VELOCITY + 4
INDEX_WHEEL_CURRENT_CONSUMPTION = INDEX_COMMANDED_VELOCITY + 2
INDEX_TENDON_LENGTH_1 = INDEX_WHEEL_CURRENT_CONSUMPTION + 4
INDEX_TENDON_LENGTH_2 = INDEX_TENDON_LENGTH_1 + 1
NUM_INPUTS = INDEX_TENDON_LENGTH_2 + 1

# Constants for output indices
NUM_POINTS = 12
INDEX_POINTS = 0
INDEX_GRAVITY = NUM_POINTS * 3
INDEX_LINEAR_VELOCITY = INDEX_GRAVITY + 3
INDEX_ANGULAR_VELOCITY = INDEX_LINEAR_VELOCITY + 3
NUM_OUTPUTS = INDEX_ANGULAR_VELOCITY + 3


class GoatShapeStateEstimation(Node):
    def __init__(self):
        super().__init__("goat_shape_state_estimation")

        # Subscribers
        imu_data_topic = self.declare_parameter("imu_data_topic", "/imu/data").get_parameter_value().string_value
        commanded_velocity_topic = (
            self.declare_parameter("commanded_velocity_topic", "/commanded_velocity").get_parameter_value().string_value
        )
        measured_velocity_topic = self.declare_parameter("measured_velocity_topic", "/measured_velocity").get_parameter_value().string_value
        current_consumption_topic = (
            self.declare_parameter("current_consumption_topic", "/current_consumption").get_parameter_value().string_value
        )
        tendon_length_1_topic = self.declare_parameter("tendon_length_topic_1", "/tendon_length_node_1/tendon_length").get_parameter_value().string_value
        tendon_length_2_topic = self.declare_parameter("tendon_length_topic_2", "/tendon_length_node_2/tendon_length").get_parameter_value().string_value

        self.imu_data_subscription = self.create_subscription(Imu, imu_data_topic, self.imu_data_callback, 10)
        self.commanded_velocity_subscription = self.create_subscription(
            Float32MultiArray, commanded_velocity_topic, self.commanded_velocity_callback, 10
        )
        self.measured_velocity_subscription = self.create_subscription(
            Float32MultiArray, measured_velocity_topic, self.measured_velocity_callback, 10
        )
        self.current_consumption_subscription = self.create_subscription(
            Float32MultiArray, current_consumption_topic, self.current_consumption_callback, 10
        )
        self.tendon_length_1_subscription = self.create_subscription(Float32, tendon_length_1_topic, self.tendon_length_1_callback, 10)
        self.tendon_length_2_subscription = self.create_subscription(Float32, tendon_length_2_topic, self.tendon_length_2_callback, 10)

        # Model initialization
        model_path = (
            self.declare_parameter("model_path", "/colcon_ws/src/goat_shape_state_estimation/models/latest.pt").get_parameter_value().string_value
        )
        self._model: torch.ScriptModule = torch.jit.load(model_path, map_location="cpu")
        self._input: torch.Tensor = torch.zeros(NUM_INPUTS, dtype=torch.float32, device="cpu")
        self._output: torch.Tensor = torch.zeros(NUM_OUTPUTS, dtype=torch.float32, device="cpu")

        # Publishers
        frame_points_topic = self.declare_parameter("frame_points_topic", "/frame_points").get_parameter_value().string_value
        gravity_vector_topic = self.declare_parameter("gravity_vector_topic", "/gravity_vector").get_parameter_value().string_value
        estimated_twist_topic = self.declare_parameter("estimated_twist", "/estimated_twist").get_parameter_value().string_value

        self.frame_points_publisher = self.create_publisher(Float32MultiArray, frame_points_topic, 10)
        self.gravity_vector_publisher = self.create_publisher(Float32MultiArray, gravity_vector_topic, 10)
        self.estimated_twist_publisher = self.create_publisher(Twist, estimated_twist_topic, 10)

        timer_period = 0.05  # seconds -> 20Hz
        self.shape_state_estimation_timer = self.create_timer(timer_period, self.shape_state_estimation_callback)
        self.get_logger().info(f"[goat_shape_state_estimation] is up and running.")

    def imu_data_callback(self, msg: Imu):
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        robot_rot_orientation_quat = torch.tensor(quat, dtype=torch.float32, device="cpu")
        robot_rot_orientation_rotmat = roma.unitquat_to_rotmat(robot_rot_orientation_quat)
        self._input[:INDEX_IMU_ANGULAR_VELOCITY] = robot_rot_orientation_rotmat[:, 2]  # this is essentially rot_mat * [0 0 1].T
        self._input[INDEX_IMU_ANGULAR_VELOCITY:INDEX_LINEAR_ACCELERATION] = torch.tensor(gyro, dtype=torch.float32)
        self._input[INDEX_LINEAR_ACCELERATION:INDEX_MEASURED_WHEEL_VELOCITY] = torch.tensor(accel, dtype=torch.float32)

    def commanded_velocity_callback(self, msg: Float32MultiArray):
        self._input[INDEX_COMMANDED_VELOCITY:INDEX_WHEEL_CURRENT_CONSUMPTION] = torch.tensor(msg.data, dtype=torch.float32)

    def measured_velocity_callback(self, msg: Float32MultiArray):
        self._input[INDEX_MEASURED_WHEEL_VELOCITY:INDEX_COMMANDED_VELOCITY] = torch.tensor(msg.data, dtype=torch.float32)

    def current_consumption_callback(self, msg: Float32MultiArray):
        self._input[INDEX_WHEEL_CURRENT_CONSUMPTION:INDEX_TENDON_LENGTH_1] = torch.tensor(msg.data, dtype=torch.float32)

    def tendon_length_1_callback(self, msg: Float32):
        self._input[INDEX_TENDON_LENGTH_1] = msg.data

    def tendon_length_2_callback(self, msg: Float32):
        self._input[INDEX_TENDON_LENGTH_2] = msg.data

    def shape_state_estimation_callback(self):
        self._output[:] = self._model(self._input.view(1, 1, -1)).view(-1)
        output_numpy = self._output.cpu().numpy()

        frame_points_msg = Float32MultiArray()
        gravity_vector_msg = Float32MultiArray()
        estimated_twist_msg = Twist()

        frame_points_msg.data = output_numpy[:INDEX_GRAVITY].tolist()
        gravity_vector_msg.data = output_numpy[INDEX_GRAVITY:INDEX_LINEAR_VELOCITY].tolist()
        linear_velocity = output_numpy[INDEX_LINEAR_VELOCITY:INDEX_ANGULAR_VELOCITY].tolist()
        angular_velocity = output_numpy[INDEX_ANGULAR_VELOCITY:].tolist()
        estimated_twist_msg.linear.x = linear_velocity[0]
        estimated_twist_msg.linear.y = linear_velocity[1]
        estimated_twist_msg.linear.z = linear_velocity[2]
        estimated_twist_msg.angular.x = angular_velocity[0]
        estimated_twist_msg.angular.y = angular_velocity[1]
        estimated_twist_msg.angular.z = angular_velocity[2]

        self.frame_points_publisher.publish(frame_points_msg)
        self.gravity_vector_publisher.publish(gravity_vector_msg)
        self.estimated_twist_publisher.publish(estimated_twist_msg)


def main(args=None):
    rclpy.init(args=args)
    goat_controller_node = GoatShapeStateEstimation()
    rclpy.spin(goat_controller_node)
    goat_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
