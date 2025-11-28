import numpy as np
from racetrack import RaceTrack


# inspired by pure pursuit formula lol 


# all parameters were tuned by a script (not included here) that ran multiple simulations with different parameters
class ControllerParams:
    """All of the controller's parameters so that I can tune easily"""

    # parameters for determining our speed
    CURVATURE_SMOOTHING_WINDOW = 2
    MAX_SAFE_ACCEL = 10.0  # caps our car's acceleration (we noticed at high speeds / accelerations car is more likely to go off track)
    MIN_SAFE_SPEED = -10.0 # caps our car's speed
    MAX_SAFE_SPEED = 45.0 
    
    # parameters for determining lookahead distance
    LOOKAHEAD_MIN = 6.0
    LOOKAHEAD_MAX = 22.0
    LOOKAHEAD_VELOCITY_GAIN = 0.4
    LOOKAHEAD_BASE_OFFSET = 5.0

    # controller gains
    VELOCITY_KP = 5.0
    STEERING_KP = 8.0
    

PARAMS = ControllerParams()
safe_speed_calculation = None 

def precompute(centerline, N): 
    global safe_speed_calculation
    # Pre-compute speed profile based on path curvature

    curvatures = np.zeros(N)
    for i in range(N):
        total_angle_change = 0
        total_dist = 0

        # commpare how much our angle and distance changes some amount of points 
        # this determines how curvy our path is, more curvy = slower speed to reduce track violations 
        
        for j in range(-PARAMS.CURVATURE_SMOOTHING_WINDOW, PARAMS.CURVATURE_SMOOTHING_WINDOW):
            idx = (i + j) % N
            p0 = centerline[(idx - 1) % N]
            p1 = centerline[idx]
            p2 = centerline[(idx + 1) % N]

            # calculate the difference between consecutive points
            v1 = p1 - p0
            v2 = p2 - p1

            # Calculate angle change between vectors
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            dangle = np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1))

            total_angle_change += abs(dangle)
            total_dist += np.linalg.norm(p2 - p1)

        # calculate curvature as angle change over distance
        curvatures[i] = total_angle_change / (total_dist + 1e-6) # avoid div by 0

    # slow down for high curvature
    profile = np.sqrt(PARAMS.MAX_SAFE_ACCEL / (curvatures + 1e-6)) # avoid div by 0
    # grab how fast we should be going at every point on the racetrack
    safe_speed_calculation = np.clip(profile, PARAMS.MIN_SAFE_SPEED, PARAMS.MAX_SAFE_SPEED)

def lower_controller(state, desired, parameters):
    """
    Lower-level controller that tracks desired steering angle and velocity.
    """
    state = np.asarray(state)
    desired = np.asarray(desired)
    parameters = np.asarray(parameters)

    steer_angle = state[2]
    vel = state[3]
    
    target_steer, target_vel = desired

    # proportional controller for velocity
    accel = PARAMS.VELOCITY_KP * (target_vel - vel)
    accel = np.clip(accel, parameters[8], parameters[10])

    # proportional controller for steering
    steer_rate = PARAMS.STEERING_KP * (target_steer - steer_angle)
    steer_rate = np.clip(steer_rate, parameters[7], parameters[9])

    return np.array([steer_rate, accel])


def controller(state, parameters, racetrack: RaceTrack):
    global safe_speed_calculation

    state = np.asarray(state)
    parameters = np.asarray(parameters)

    x, y, delta, v, phi = state
    car_pos = np.array([x, y])

    wheelbase = parameters[0]
    min_steer = parameters[1]
    max_steer = parameters[4]

    PARAMS.MAX_SAFE_SPEED = min(PARAMS.MAX_SAFE_SPEED, parameters[5])
    PARAMS.MIN_SAFE_SPEED = max(PARAMS.MIN_SAFE_SPEED, parameters[2])
    PARAMS.MAX_SAFE_ACCEL = min(PARAMS.MAX_SAFE_ACCEL, parameters[10])

    centerline = racetrack.centerline
    N = len(centerline)

    if(safe_speed_calculation is None):
        precompute(centerline, N) # just do this once so we are faster 


    # find closest track point
    dist_sq = np.sum((centerline - car_pos)**2, axis=1)
    closest_idx = np.argmin(dist_sq)

    # lookahead based on speed, when we used fixed offset of 1 or 2 lookahead we overshot 
    # lookahead based on speed prevents overshooting when calculating our speed and curvature of where we are currently 
    # also if we are curved our speed slows so we want to look at nearer points instead of points continuously fixed distances away 
    lookahead_dist = PARAMS.LOOKAHEAD_VELOCITY_GAIN * abs(v) + PARAMS.LOOKAHEAD_BASE_OFFSET
    lookahead_dist = np.clip(lookahead_dist, PARAMS.LOOKAHEAD_MIN, PARAMS.LOOKAHEAD_MAX)

    # iterate forward to find target point
    current_lookahead = 0.0
    target_idx = closest_idx
    
    # get the point closest to our lookahead point in distance 
    for i in range(1, N):
        curr = (closest_idx + i - 1) % N
        nex = (closest_idx + i) % N
        
        segment_len = np.linalg.norm(centerline[nex] - centerline[curr])
        current_lookahead += segment_len
        
        if current_lookahead >= lookahead_dist:
            target_idx = nex
            break

    # basically the point on the centerline that is closest to the car
    goal_point = centerline[target_idx]

    # Calculate steering:
    #
    # calculation on how to get to goal point
    dx_goal = goal_point[0] - x
    dy_goal = goal_point[1] - y

    # get distance to goal 
    dist_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)

    # avoid oscillations in the case where we're already really close to the goal. Just don't steer at all; i.e, we accept some amount of error to avoid overcorrection.
    if dist_to_goal < 0.1:
        target_steer = 0.0
    else:
        # Calculate the angle from the car to the goal
        angle_to_goal = np.arctan2(dy_goal, dx_goal)

        # Calculate the angular error between the car's current heading
        change_in_steering_angle = angle_to_goal - phi
        
        # normalize to [-pi, pi]
        change_in_steering_angle = (change_in_steering_angle + np.pi) % (2 * np.pi) - np.pi   

        #  Apply the Pure Pursuit Formula: delta = arctan(2L * sin(alpha) / Ld) attained from google 
        #    L = wheelbase, alpha = angular_error, Ld = dist_to_goal
        # target steering angle is 
        target_steer = np.arctan(2.0 * wheelbase * np.sin(change_in_steering_angle) / dist_to_goal)

    # Clamp it
    target_steer = np.clip(target_steer, min_steer, max_steer)

    # get velocity target from profile of how much we're looking ahead
    target_vel = safe_speed_calculation[target_idx] 

    return np.array([target_steer, target_vel])
