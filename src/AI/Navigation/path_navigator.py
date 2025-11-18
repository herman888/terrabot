"""
Terrabot Navigation Module
Handles path planning, distance estimation, and movement control
Interfaces with C++ actuation system for precise control
"""

import numpy as np
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from enum import Enum
import heapq
from collections import deque


class NavigationMode(Enum):
    """Navigation operation modes"""
    IDLE = 0
    NAVIGATING = 1
    SNOW_REMOVAL = 2
    OBSTACLE_AVOIDANCE = 3
    RETURNING_HOME = 4


@dataclass
class Waypoint:
    """Represents a navigation waypoint"""
    x: float
    y: float
    heading: float  # Target heading in radians
    tolerance: float = 0.2  # Acceptable distance from waypoint in meters
    
    def distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance to a point"""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


@dataclass
class RobotPose:
    """Current robot position and orientation"""
    x: float
    y: float
    theta: float  # Heading in radians
    timestamp: float
    
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.timestamp = time.time()


@dataclass
class NavigationCommand:
    """Commands sent to actuation system"""
    linear_velocity: float   # m/s forward/backward
    angular_velocity: float  # rad/s rotation
    snow_removal_active: bool
    timestamp: float
    
    def to_dict(self):
        return asdict(self)


class DistanceCalculator:
    """
    Handles various distance and spatial calculations for navigation
    """
    
    @staticmethod
    def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    @staticmethod
    def manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Manhattan distance between two points"""
        return abs(x2 - x1) + abs(y2 - y1)
    
    @staticmethod
    def angle_between_points(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate angle from point 1 to point 2
        Returns angle in radians [-pi, pi]
        """
        return math.atan2(y2 - y1, x2 - x1)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def angle_difference(target: float, current: float) -> float:
        """
        Calculate smallest angle difference between target and current
        Returns signed difference in radians [-pi, pi]
        """
        diff = target - current
        return DistanceCalculator.normalize_angle(diff)
    
    @staticmethod
    def point_to_line_distance(px: float, py: float, 
                               x1: float, y1: float, 
                               x2: float, y2: float) -> float:
        """Calculate perpendicular distance from point to line segment"""
        # Line segment length
        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if line_len_sq == 0:
            return DistanceCalculator.euclidean_distance(px, py, x1, y1)
        
        # Parameter t represents projection of point onto line
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
        
        # Closest point on line segment
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return DistanceCalculator.euclidean_distance(px, py, proj_x, proj_y)


class PIDController:
    """
    PID controller for smooth navigation control
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-1.0, 1.0)):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Min and max output values
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()
    
    def update(self, error: float, dt: Optional[float] = None) -> float:
        """
        Calculate PID control output
        
        Args:
            error: Current error (setpoint - measured)
            dt: Time step (calculated if None)
            
        Returns:
            Control output
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.previous_time
        
        # Avoid division by zero
        if dt <= 0:
            dt = 0.01
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        # Store for next iteration
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()


class PathPlanner:
    """
    A* path planning algorithm for optimal path generation
    """
    
    def __init__(self, grid_resolution: float = 0.5):
        """
        Initialize path planner
        
        Args:
            grid_resolution: Size of grid cells in meters
        """
        self.grid_resolution = grid_resolution
        self.obstacle_map = None  # TODO: Implement obstacle map
        
    def plan_path(self, start: Tuple[float, float], 
                  goal: Tuple[float, float]) -> List[Waypoint]:
        """
        Plan path from start to goal using A* algorithm
        
        Args:
            start: Starting (x, y) position
            goal: Goal (x, y) position
            
        Returns:
            List of waypoints forming the path
        """
        print(f"[PathPlanner] Planning path from {start} to {goal}")
        
        # TODO: Implement full A* algorithm
        # For now, create simple straight-line path with waypoints
        
        waypoints = []
        
        # Calculate number of waypoints based on distance
        distance = DistanceCalculator.euclidean_distance(start[0], start[1], goal[0], goal[1])
        num_waypoints = max(2, int(distance / self.grid_resolution))
        
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            heading = DistanceCalculator.angle_between_points(start[0], start[1], goal[0], goal[1])
            
            waypoints.append(Waypoint(x, y, heading))
        
        print(f"[PathPlanner] Generated path with {len(waypoints)} waypoints")
        return waypoints
    
    def smooth_path(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """
        Smooth path to reduce sharp turns
        
        Args:
            waypoints: Original waypoint list
            
        Returns:
            Smoothed waypoint list
        """
        # TODO: Implement path smoothing (e.g., Bezier curves, splines)
        return waypoints


class PurePursuitController:
    """
    Pure Pursuit path following controller
    Calculates steering commands to follow a path
    """
    
    def __init__(self, lookahead_distance: float = 1.0, 
                 max_linear_velocity: float = 0.5,
                 max_angular_velocity: float = 1.0):
        """
        Initialize Pure Pursuit controller
        
        Args:
            lookahead_distance: Distance ahead to look for path tracking
            max_linear_velocity: Maximum forward speed in m/s
            max_angular_velocity: Maximum rotation speed in rad/s
        """
        self.lookahead_distance = lookahead_distance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
    
    def find_lookahead_point(self, path: List[Waypoint], 
                            pose: RobotPose) -> Optional[Waypoint]:
        """
        Find point on path at lookahead distance
        
        Args:
            path: List of waypoints
            pose: Current robot pose
            
        Returns:
            Lookahead waypoint or None if path is complete
        """
        for waypoint in path:
            distance = waypoint.distance_to(pose.x, pose.y)
            if distance >= self.lookahead_distance:
                return waypoint
        
        # Return last waypoint if all are within lookahead distance
        return path[-1] if path else None
    
    def compute_control(self, lookahead_point: Waypoint, 
                       pose: RobotPose) -> NavigationCommand:
        """
        Compute control commands using Pure Pursuit algorithm
        
        Args:
            lookahead_point: Target point to pursue
            pose: Current robot pose
            
        Returns:
            Navigation command
        """
        # Calculate angle to lookahead point
        target_angle = DistanceCalculator.angle_between_points(
            pose.x, pose.y, lookahead_point.x, lookahead_point.y)
        
        # Calculate angular error
        angle_error = DistanceCalculator.angle_difference(target_angle, pose.theta)
        
        # Calculate curvature for Pure Pursuit
        distance = lookahead_point.distance_to(pose.x, pose.y)
        
        if distance < 0.01:
            # Very close to target
            return NavigationCommand(0.0, 0.0, False, time.time())
        
        # Pure Pursuit curvature formula
        curvature = (2 * math.sin(angle_error)) / distance
        
        # Calculate velocities
        angular_velocity = curvature * self.max_linear_velocity
        angular_velocity = max(-self.max_angular_velocity, 
                             min(self.max_angular_velocity, angular_velocity))
        
        # Reduce linear velocity when turning sharply
        linear_velocity = self.max_linear_velocity * (1 - abs(angular_velocity) / self.max_angular_velocity)
        
        return NavigationCommand(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            snow_removal_active=False,
            timestamp=time.time()
        )


class NavigationController:
    """
    High-level navigation controller integrating all components
    """
    
    def __init__(self):
        """Initialize navigation controller"""
        self.pose = RobotPose()
        self.mode = NavigationMode.IDLE
        self.current_path = []
        self.current_waypoint_index = 0
        
        # Initialize sub-controllers
        self.path_planner = PathPlanner(grid_resolution=0.5)
        self.pursuit_controller = PurePursuitController(
            lookahead_distance=1.0,
            max_linear_velocity=0.5,
            max_angular_velocity=1.0
        )
        
        # PID controllers for fine-tuning
        self.linear_pid = PIDController(kp=1.0, ki=0.01, kd=0.1, 
                                       output_limits=(-0.5, 0.5))
        self.angular_pid = PIDController(kp=2.0, ki=0.02, kd=0.2,
                                        output_limits=(-1.0, 1.0))
        
        print("[NavigationController] Initialized")
    
    def update_pose(self, x: float, y: float, theta: float):
        """Update robot pose from odometry/sensors"""
        self.pose = RobotPose(x, y, theta)
    
    def navigate_to_target(self, target_x: float, target_y: float) -> bool:
        """
        Plan and start navigation to target location
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            
        Returns:
            True if path planning successful
        """
        start = (self.pose.x, self.pose.y)
        goal = (target_x, target_y)
        
        # Plan path
        self.current_path = self.path_planner.plan_path(start, goal)
        
        if not self.current_path:
            print("[NavigationController] Path planning failed")
            return False
        
        # Smooth path
        self.current_path = self.path_planner.smooth_path(self.current_path)
        
        self.current_waypoint_index = 0
        self.mode = NavigationMode.NAVIGATING
        
        print(f"[NavigationController] Navigation started to ({target_x}, {target_y})")
        return True
    
    def update(self) -> NavigationCommand:
        """
        Main update loop - compute navigation commands
        
        Returns:
            Navigation command for actuation system
        """
        if self.mode == NavigationMode.IDLE:
            return NavigationCommand(0.0, 0.0, False, time.time())
        
        if not self.current_path:
            self.mode = NavigationMode.IDLE
            return NavigationCommand(0.0, 0.0, False, time.time())
        
        # Check if current waypoint is reached
        current_waypoint = self.current_path[self.current_waypoint_index]
        distance_to_waypoint = current_waypoint.distance_to(self.pose.x, self.pose.y)
        
        if distance_to_waypoint < current_waypoint.tolerance:
            self.current_waypoint_index += 1
            
            if self.current_waypoint_index >= len(self.current_path):
                # Goal reached
                print("[NavigationController] Goal reached!")
                self.mode = NavigationMode.IDLE
                return NavigationCommand(0.0, 0.0, False, time.time())
        
        # Get remaining path
        remaining_path = self.current_path[self.current_waypoint_index:]
        
        # Find lookahead point
        lookahead_point = self.pursuit_controller.find_lookahead_point(
            remaining_path, self.pose)
        
        if not lookahead_point:
            self.mode = NavigationMode.IDLE
            return NavigationCommand(0.0, 0.0, False, time.time())
        
        # Compute control command
        command = self.pursuit_controller.compute_control(lookahead_point, self.pose)
        
        # Activate snow removal if in snow removal mode
        if self.mode == NavigationMode.SNOW_REMOVAL:
            command.snow_removal_active = True
        
        return command
    
    def emergency_stop(self):
        """Stop all navigation immediately"""
        self.mode = NavigationMode.IDLE
        self.current_path = []
        print("[NavigationController] EMERGENCY STOP")
    
    def get_distance_to_goal(self) -> float:
        """Get remaining distance to goal"""
        if not self.current_path:
            return 0.0
        
        goal = self.current_path[-1]
        return goal.distance_to(self.pose.x, self.pose.y)
    
    def get_status(self) -> dict:
        """Get current navigation status"""
        return {
            "mode": self.mode.name,
            "pose": {"x": self.pose.x, "y": self.pose.y, "theta": self.pose.theta},
            "waypoints_remaining": len(self.current_path) - self.current_waypoint_index,
            "distance_to_goal": self.get_distance_to_goal()
        }


def main():
    """Test navigation system"""
    print("=" * 50)
    print("  Terrabot Navigation System")
    print("  Path Planning & Distance Estimation")
    print("=" * 50)
    
    # Create navigation controller
    nav = NavigationController()
    
    # Simulate navigation
    nav.update_pose(0.0, 0.0, 0.0)
    nav.navigate_to_target(5.0, 5.0)
    
    # Simulation loop
    for i in range(100):
        # Update position (simulated)
        nav.update_pose(i * 0.05, i * 0.05, math.pi / 4)
        
        # Get command
        cmd = nav.update()
        
        if i % 10 == 0:
            status = nav.get_status()
            print(f"Step {i}: Mode={status['mode']}, "
                  f"Distance to goal={status['distance_to_goal']:.2f}m")
            print(f"  Command: v={cmd.linear_velocity:.2f}, "
                  f"w={cmd.angular_velocity:.2f}")
        
        if nav.mode == NavigationMode.IDLE:
            break
        
        time.sleep(0.05)


if __name__ == "__main__":
    main()
