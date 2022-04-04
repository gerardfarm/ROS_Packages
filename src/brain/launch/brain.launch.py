import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'node_prefix',
            default_value=[launch.substitutions.EnvironmentVariable('USER'), '_'],
            description='Prefix for node names'),
        # Localization Part
        launch_ros.actions.Node(
            package='localization', executable='acc_publisher', output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'acc_publisher']),
        launch_ros.actions.Node(
            package='localization', executable='gyro_publisher', output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'gyro_publisher']),
        launch_ros.actions.Node(
            package='localization', executable='gps_publisher', output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'gps_publisher']),
        launch_ros.actions.Node(
            package='localization', executable='localize_robot', output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'localize_robot']),
        # Perception Part
        launch_ros.actions.Node(
            package='perception', executable='cam_publisher', output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'cam_publisher']),
        launch_ros.actions.Node(
            package='perception', executable='detect_objects', output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'detect_objects']),
        # Brain Part
        launch_ros.actions.Node(
            package='brain', executable='brain_subscriber', output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'brain_subscriber']),
    ])