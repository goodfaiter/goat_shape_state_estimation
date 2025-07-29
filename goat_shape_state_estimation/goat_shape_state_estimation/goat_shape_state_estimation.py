import torch
from torch import ScriptModule, Tensor
import roma
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, Float32


INDEX_IMU_ANGULAR_VELOCITY = 3
INDEX_LINEAR_ACCELERATION = INDEX_IMU_ANGULAR_VELOCITY + 3
INDEX_MEASURED_WHEEL_VELOCITY = INDEX_LINEAR_ACCELERATION + 3
INDEX_COMMANDED_VELOCITY = INDEX_MEASURED_WHEEL_VELOCITY + 4
INDEX_WHEEL_CURRENT_CONSUMPTION = INDEX_COMMANDED_VELOCITY + 2
INDEX_TENDON_LENGTH_1 = INDEX_WHEEL_CURRENT_CONSUMPTION + 4
INDEX_TENDON_LENGTH_2 = INDEX_TENDON_LENGTH_1 + 1
NUM_INPUTS = INDEX_TENDON_LENGTH_2 + 1
NUM_POINTS = 12
INDEX_POINTS = 0
INDEX_GRAVITY = NUM_POINTS * 3
INDEX_LINEAR_VELOCITY = INDEX_GRAVITY + 3
INDEX_ANGULAR_VELOCITY = INDEX_LINEAR_VELOCITY + 3
NUM_OUTPUTS = INDEX_ANGULAR_VELOCITY + 3

class GoatShapeStateEstimation(Node):
    def __init__(self):
        super().__init__('goat_shape_state_estimation')

        # Subscribers
        imu_data_topic = self.declare_parameter("imu_data_topic", "/imu/data").get_parameter_value().string_value
        commanded_velocity_topic = self.declare_parameter("commanded_velocity_topic", "/commanded_velocity").get_parameter_value().string_value
        measured_velocity_topic = self.declare_parameter("measured_velocity_topic", "/measured_velocity").get_parameter_value().string_value
        current_consumption_topic = self.declare_parameter("current_consumption_topic", "/current_consumption").get_parameter_value().string_value
        tendon_length_1_topic = self.declare_parameter("tendon_length_topic_1", "/tendon_length_node_1").get_parameter_value().string_value
        tendon_length_2_topic = self.declare_parameter("tendon_length_topic_2", "/tendon_length_node_2").get_parameter_value().string_value

        self.imu_data_subscription = self.create_subscription(Imu, imu_data_topic, self.imu_data_callback, 10)
        self.commanded_velocity_subscription = self.create_subscription(Float32MultiArray, commanded_velocity_topic, self.commanded_velocity_callback, 10)
        self.measured_velocity_subscription = self.create_subscription(Float32MultiArray, measured_velocity_topic, self.measured_velocity_callback, 10)
        self.current_consumption_subscription = self.create_subscription(Float32MultiArray, current_consumption_topic, self.current_consumption_callback, 10)
        self.tendon_length_1_subscription = self.create_subscription(Float32MultiArray, tendon_length_1_topic, self.tendon_length_1_callback, 10)
        self.tendon_length_2_subscription = self.create_subscription(Float32MultiArray, tendon_length_2_topic, self.tendon_length_2_callback, 10)

        robot_rot_orientation_quat = torch.zeros(4, dtype=torch.float, device='cpu')
        model_path = self.declare_parameter('model_path', 'colon_ws/goat_shape_state_estimation/models/latest.pt').get_parameter_value().string
        self._model: ScriptModule = torch.jit.load(model_path, map_location="cpu")
        self._input: Tensor = torch.zeroes(NUM_INPUTS, dtype=torch.float, device='cpu')
        self._output: Tensor = torch.zeroes(NUM_OUTPUTS, dtype=torch.float, device='cpu')

        # Publishers
        frame_points_topic = self.declare_parameter("frame_points_topic", "/frame_points").get_parameter_value().string_value
        gravity_vector_topic = self.declare_parameter("gravity_vector_topic", "/gravity_vector").get_parameter_value().string_value
        linear_velocity_topic = self.declare_parameter("linear_velocity_topic", "/linear_velocity").get_parameter_value().string_value
        angular_velocity_topic = self.declare_parameter("angular_velocity_topic", "/angular_velocity").get_parameter_value().string_value

        self.frame_points_publisher = self.create_publisher(Float32MultiArray, frame_points_topic, 10)
        self.gravity_vector_publisher = self.create_publisher(Float32MultiArray, gravity_vector_topic, 10)
        self.linear_velocity_publisher = self.create_publisher(Float32, linear_velocity_topic, 10)
        self.angular_velocity_publisher = self.create_publisher(Float32, angular_velocity_topic, 10)

        timer_period = 0.05  # seconds -> 20Hz
        self.shape_state_estimation_timer = self.create_timer(timer_period, self.shape_state_estimation_callback)

    def imu_data_callback(self, msg: Imu):
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        robot_rot_orientation_quat = torch.tensor(quat, dtype=torch.float, device=self.device)
        robot_rot_orientation_rotmat = roma.unitquat_to_rotmat(robot_rot_orientation_quat)
        self._input[:INDEX_GRAVITY] = robot_rot_orientation_rotmat[:, 2]  # this is essentially rot_mat * [0 0 1].T
        self._input[INDEX_IMU_ANGULAR_VELOCITY:INDEX_LINEAR_ACCELERATION] = gyro
        self._input[INDEX_LINEAR_ACCELERATION:INDEX_MEASURED_WHEEL_VELOCITY] = accel

    def commanded_velocity_callback(self, msg: Float32MultiArray):
        self._input[INDEX_MEASURED_WHEEL_VELOCITY:INDEX_COMMANDED_VELOCITY] = msg.data

    def measured_velocity_callback(self, msg: Float32MultiArray):
        self._input[INDEX_COMMANDED_VELOCITY:INDEX_WHEEL_CURRENT_CONSUMPTION] = msg.data

    def current_consumption_callback(self, msg: Float32MultiArray):
        self._input[INDEX_WHEEL_CURRENT_CONSUMPTION:INDEX_TENDON_LENGTH_1] = msg.data

    def current_consumption_callback(self, msg: Float32):
        self._input[INDEX_TENDON_LENGTH_1:INDEX_TENDON_LENGTH_2] = msg.data

    def current_consumption_callback(self, msg: Float32):
        self._input[INDEX_TENDON_LENGTH_2:] = msg.data

    def shape_state_estimation_callback(self):
        self._output[:] = self._model(self._input.view(1, 1, -1)).view(-1)
        output_numpy = output.cpu().numpy()

        frame_points_msg = Float32MultiArray()
        gravity_vector_msg = Float32MultiArray()
        linear_velocity_msg = Float32MultiArray()
        angular_velocity_msg = Float32MultiArray()
        frame_points_msg.data = output_numpy[:INDEX_GRAVITY]
        gravity_vector_msg.data = output_numpy[INDEX_GRAVITY:INDEX_LINEAR_VELOCITY]
        linear_velocity_msg.data = output_numpy[INDEX_LINEAR_VELOCITY:INDEX_ANGULAR_VELOCITY]
        angular_velocity_msg.data = output_numpy[INDEX_ANGULAR_VELOCITY:]
        self.frame_points_publisher.publish(frame_points_msg)
        self.gravity_vector_publisher.publish(gravity_vector_msg)
        self.linear_velocity_publisher.publish(linear_velocity_msg)
        self.angular_velocity_publisher.publish(angular_velocity_msg)


def main(args=None):
    rclpy.init(args=args)
    goat_controller_node = GoatShapeStateEstimation()
    rclpy.spin(goat_controller_node)
    goat_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# Load the traced model
traced_model = torch.jit.load("/workspace/data/output/latest_lstm_model.pt", map_location="cpu")
# traced_model.reset()
# data = pd.read_parquet("/workspace/data/2025_06_04/2025_06_04_16_24_22_goat_training.parquet") # c -> s -> c
# data = pd.read_parquet("/workspace/data/2025_06_04/2025_06_04_15_52_28_goat_training.parquet") # yaw in circle mode
data = pd.read_parquet("/workspace/data/2025_07_21/rosbag2_2025_07_21-14_16_19_goat_training.parquet") # yaw in circle mode
# data = pd.read_parquet("/workspace/data/2025_07_21/rosbag2_2025_07_22-08_50_04_goat_training.parquet") # drive in rover
# data = pd.read_parquet("/workspace/data/2025_06_04/2025_06_04_15_50_40_goat_training.parquet") # rov -> c

device = device = torch.device("cpu")

data_processor_goat = DataProcessorGoat(device)
inputs = data_processor_goat.process_input_data(data)
targets = data_processor_goat.process_output_data(data)

num_points = 12
estimated_points = torch.zeros([inputs.shape[0], 3 * num_points], dtype=torch.float, device=torch.device('cpu'))
estimated_gravity = torch.zeros([inputs.shape[0], 3], dtype=torch.float, device=torch.device('cpu'))
grav_index = num_points * 3
estimated_velocities = torch.zeros([inputs.shape[0], 6], dtype=torch.float, device=torch.device('cpu'))
vel_index = grav_index + 3

for i in range(inputs.shape[0] - 20):
    # sample_input = inputs[i: i + 20, :].unsqueeze(0)
    sample_input = inputs[i, :].unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = traced_model(sample_input)
