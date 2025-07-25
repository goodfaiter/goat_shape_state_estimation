import torch
import pandas as pd
import numpy as np
from helpers.data_processer import DataProcessorGoat
from helpers.visualizer import visualize_3d_spline, plot_velocity_comparison


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

from .dynamixel_controller import Dynamixel



        num_data = data["/imu/data/orientation_w"].size
        data_tensor = torch.zeros([num_data, self.input_shape], dtype=torch.float, device=self.device)
        robot_rot_orientation_quat = torch.zeros([num_data, 4], dtype=torch.float, device=self.device)
        robot_rot_orientation_quat[:, 0] = torch.tensor(data["/imu/data/orientation_x"], dtype=torch.float, device=self.device)
        robot_rot_orientation_quat[:, 1] = torch.tensor(data["/imu/data/orientation_y"], dtype=torch.float, device=self.device)
        robot_rot_orientation_quat[:, 2] = torch.tensor(data["/imu/data/orientation_z"], dtype=torch.float, device=self.device)
        robot_rot_orientation_quat[:, 3] = torch.tensor(data["/imu/data/orientation_w"], dtype=torch.float, device=self.device)
        robot_rot_orientation_rotmat = roma.unitquat_to_rotmat(robot_rot_orientation_quat)
        data_tensor[:, 0:3] = robot_rot_orientation_rotmat[:, :, 2]  # this is essentially rot_mat * [0 0 1].T
        # data_tensor[:, 0:3] = ema_2d_optimized(data_tensor[:, 0:3])

        data_tensor[:, 3] = torch.tensor(data["/imu/data/angular_velocity_x"], dtype=torch.float, device=self.device)
        data_tensor[:, 4] = torch.tensor(data["/imu/data/angular_velocity_y"], dtype=torch.float, device=self.device)
        data_tensor[:, 5] = torch.tensor(data["/imu/data/angular_velocity_z"], dtype=torch.float, device=self.device)
        # data_tensor[:, 3:6] = ema_2d_optimized(data_tensor[:, 3:6])

        data_tensor[:, 6] = torch.tensor(data["/imu/data/linear_acceleration_x"], dtype=torch.float, device=self.device)
        data_tensor[:, 7] = torch.tensor(data["/imu/data/linear_acceleration_y"], dtype=torch.float, device=self.device)
        data_tensor[:, 8] = torch.tensor(data["/imu/data/linear_acceleration_z"], dtype=torch.float, device=self.device)
        # data_tensor[:, 6:9] = ema_2d_optimized(data_tensor[:, 6:9])

        data_tensor[:, 9] = torch.tensor(data["/measured_velocity/data_0"], dtype=torch.float, device=self.device)
        data_tensor[:, 10] = torch.tensor(data["/measured_velocity/data_1"], dtype=torch.float, device=self.device)
        data_tensor[:, 11] = torch.tensor(data["/measured_velocity/data_2"], dtype=torch.float, device=self.device)
        data_tensor[:, 12] = torch.tensor(data["/measured_velocity/data_3"], dtype=torch.float, device=self.device)

        data_tensor[:, 13] = torch.tensor(data["/commanded_velocity/data_0"], dtype=torch.float, device=self.device)
        data_tensor[:, 14] = torch.tensor(data["/commanded_velocity/data_1"], dtype=torch.float, device=self.device)

        data_tensor[:, 15] = torch.tensor(data["/current_consumption/data_0"], dtype=torch.float, device=self.device)
        data_tensor[:, 16] = torch.tensor(data["/current_consumption/data_1"], dtype=torch.float, device=self.device)
        data_tensor[:, 17] = torch.tensor(data["/current_consumption/data_2"], dtype=torch.float, device=self.device)
        data_tensor[:, 18] = torch.tensor(data["/current_consumption/data_3"], dtype=torch.float, device=self.device)

        data_tensor[:, 19] = torch.tensor(data["/tendon_length_node_1/tendon_length/data"], dtype=torch.float, device=self.device)
        data_tensor[:, 20] = torch.tensor(data["/tendon_length_node_2/tendon_length/data"], dtype=torch.float, device=self.device)


class GoatShapeStateEstimation(Node):
    def __init__(self):
        super().__init__('goat_shape_state_estimation')

        imu_data_topic = self.declare_parameter("imu_data_topic", "/imu/data").get_parameter_value().string_value
        commanded_velocity_topic = self.declare_parameter("commanded_velocity_topic", "/commanded_velocity").get_parameter_value().string_value
        measured_velocity_topic = self.declare_parameter("measured_velocity_topic", "/measured_velocity").get_parameter_value().string_value
        current_consumption_topic = self.declare_parameter("current_consumption_topic", "/current_consumption").get_parameter_value().string_value
        tendon_length_1_topic = self.declare_parameter("tendon_length_topic_1", "/tendon_length_node_1").get_parameter_value().string_value
        tendon_length_2_topic = self.declare_parameter("tendon_length_topic_2", "/tendon_length_node_2").get_parameter_value().string_value

        self.imu_data_subscription = self.create_subscription(Float32MultiArray, imu_data_topic, self.imu_data_callback, 10)
        self.commanded_velocity_subscription = self.create_subscription(Float32MultiArray, commanded_velocity_topic, self.commanded_velocity_callback, 10)
        self.measured_velocity_subscription = self.create_subscription(Float32MultiArray, measured_velocity_topic, self.measured_velocity_callback, 10)
        self.current_consumption_subscription = self.create_subscription(Float32MultiArray, current_consumption_topic, self.current_consumption_callback, 10)
        self.tendon_length_1_subscription = self.create_subscription(Float32MultiArray, tendon_length_1_topic, self.tendon_length_1_callback, 10)
        self.tendon_length_2_subscription = self.create_subscription(Float32MultiArray, tendon_length_2_topic, self.tendon_length_2_callback, 10)

        self.imu_data_tensor = torch.zeroes((21), dtype=torch.float, device='cpu')


        model_path = self.declare_parameter('model_path', 'colon_ws/goat_shape_state_estimation/models/latest.pt').get_parameter_value().string
        self._model = torch.jit.load(model_path, map_location="cpu")
        self._input = torch.zeroes((1, 1, 21), dtype=torch.float, device='cpu')

        # publisher for points
        # publisher for gravity
        # publisher for lin vel
        # publisher for ang vel

        timer_period = 0.05  # seconds -> 20Hz
        self.state_timer = self.create_timer(timer_period, self.state_callback)


    def imu_data_callback(self, msg: Float32MultiArray):
        linear_velocity = 0.0
        angular_velocity = 0.0

    def state_callback(self):
        # fill out self._input
        output = self._model(self._input)
        output = output.cpu().numpy()
        # publish


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
