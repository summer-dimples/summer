# model with density control
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import matplotlib.tri as tri

# dict to store different driver type(can also be used for reaction/...)
# make sure sapwn_weight sum up to 1
DRIVER_TYPES = {
    0: { # Lorry
        'vmax': 3,
        'overtake_rate': 0,    
        'return_rate': 1,
        'spawn_weight': 0.2,
        'boost_chance': 0.3,    # this can be used to simulate the need time for boost?(for example, 0.5 means average 2 timestep before boost)
        'boosted_vmax': 5,
        'random_brake_chance' : 0.3      
    },
    1: { # Regular Cars
        'vmax': 5,
        'overtake_rate': 0.5,    
        'return_rate': 0.5,     
        'spawn_weight': 0.6,
        'boost_chance': 0.3,   
        'boosted_vmax': 7,
        'random_brake_chance' : 0.3    
    },
    2: { # Fast Cars
        'vmax': 7,
        'overtake_rate': 0.7,    
        'return_rate': 0,
        'spawn_weight':0.2,
        'boost_chance': 0.3,    
        'boosted_vmax': 9,
        'random_brake_chance' : 0.3          
    },
    3: { # Fixed Obstacle
        'vmax': 0,  
        'overtake_rate': 0, 
        'return_rate': 0,    
        'spawn_weight': 0,   
        'boost_chance': 0,   
        'boosted_vmax': 0,   
        'random_brake_chance': 0
    }
}

# get the gap same lane ahead
def get_gap_ahead_same_lane(lane, pos):
    # check if the current position is valid
    if pos < 0 or pos >= len(lane):
        raise ValueError("Out of range")
    
    # find all the cars ahead, if it's there is car in next position, return 0(no gap)
    next_car_positions = np.where(lane[pos+1:] >= 0)[0]
    
    # if no car in front of it
    if len(next_car_positions) == 0:
        return 100
    
    return next_car_positions[0]

# get gaps from other lane
def get_gaps_other_lane(lanes, lane_idx, pos):
    other_lane = lanes[1 - lane_idx]
    
    if pos < 0 or pos >= len(lanes[0]):
        raise ValueError("Current position is out of range")
    
    #if there is a vehicle on a neighbouring site both return -1
    if other_lane[pos] >= 0:
        return -1, -1
    
    # Forward gap:
    next_car_positions = np.where(other_lane[pos+1:] >= 0)[0]
    
    # if no car in front of it
    if len(next_car_positions) == 0:
        forward_gap = 100
    else:
        forward_gap = next_car_positions[0]  
        
    # Backward gap: 
    prev_car_positions = np.where(other_lane[:pos] >= 0)[0]
    
    # no car at back
    if len(prev_car_positions) == 0:
        backward_gap = 100  
    else:
        backward_gap = pos - prev_car_positions[-1] - 1  
    
    return forward_gap, backward_gap

# get the gap same lane behind for boost
def get_gap_behind_same_lane(lane, pos):
    # check if the current position is valid
    if pos < 0 or pos >= len(lane):
        raise ValueError("Out of range")
    
    prev_car_positions = np.where(lane[:pos] >= 0)[0]
    
    # if no car behind
    if len(prev_car_positions) == 0:
        return 100
    
    return pos - prev_car_positions[-1] - 1

# check whether should change the lane
def should_change_lane(lanes, lane_idx, pos, driver_types, driver_types_dict=None):
    if driver_types_dict is None:
        driver_types_dict = DRIVER_TYPES
        
    if lanes[lane_idx][pos] == -1:
        return False
    
    
    driver_type = driver_types[lane_idx][pos]
    
    if driver_type == -1:
        return False

    
    forward_gap = get_gap_ahead_same_lane(lanes[lane_idx], pos)
    adj_forward_gap, adj_backward_gap = get_gaps_other_lane(lanes, lane_idx, pos)
    
    v = lanes[lane_idx][pos]
    

    driver_params = driver_types_dict[driver_type]
    overtake_rate = driver_params['overtake_rate']
    return_rate = driver_params['return_rate']
    
    # When in right lane (lane_idx = 1), check if should return to left lane
    if lane_idx == 1:  
        if (adj_forward_gap > v + 1 and      # T2
            adj_backward_gap >= 6 and           # T3
            np.random.random() < return_rate):  # T4
            return True

    # overtake
    if (forward_gap < v + 1 and                  # T1
        adj_forward_gap > v + 1 and            # T2
        adj_backward_gap >= 6 and                # T3
        np.random.random() < overtake_rate):  # T4
        return True
    
    return False 

# updatge change of lane
def change_lanes(lanes, driver_types, driver_types_dict=None):
    length = len(lanes[0])
    changes = []
    
    # record the index and position that would change lane
    for lane_idx in range(2):
        for pos in range(length):
            if should_change_lane(lanes, lane_idx, pos, driver_types, driver_types_dict):
                changes.append((lane_idx, pos))
                
    # change lane
    for lane_idx, pos in changes:
        if lanes[lane_idx][pos] >= 0:
            # get velocity and driver type
            v = lanes[lane_idx][pos]
            driver_type = driver_types[lane_idx][pos]
            
            # change lane
            lanes[lane_idx][pos] = -1
            driver_types[lane_idx][pos] = -1
            
            lanes[1-lane_idx][pos] = v
            driver_types[1-lane_idx][pos] = driver_type
    
    return lanes, driver_types, len(changes)

# update velocities for each step
def update_velocities(lanes, driver_types, random_brake_override = None, driver_types_dict=None):
    if driver_types_dict is None:
        driver_types_dict = DRIVER_TYPES
    
    boosted_cars = [np.zeros(len(lanes[0]), dtype=bool), np.zeros(len(lanes[0]), dtype=bool)]
    
    # only boost on fast lane
    for pos in range(len(lanes[1])):  
        if lanes[1][pos] >= 0:  
            driver_type = driver_types[1][pos]
            
            boosted_vmax = driver_types_dict[driver_type]['boosted_vmax']
            
            gap_ahead = get_gap_ahead_same_lane(lanes[1], pos)
            gap_behind = get_gap_behind_same_lane(lanes[1], pos)
            
            # check if need to boost
            if (gap_ahead > boosted_vmax + 1 and            # empty at front
                gap_behind < 3 and gap_behind > 0 and             # if block someone
                np.random.random() < driver_types_dict[driver_type]['boost_chance']):  
                
                boosted_cars[1][pos] = True
    
    
    for lane_idx in range(2):
        for pos in range(len(lanes[0])):
            if lanes[lane_idx][pos] >= 0:
                driver_type = driver_types[lane_idx][pos]
                v = lanes[lane_idx][pos]
                
                base_vmax = driver_types_dict[driver_type]['vmax']
                
                real_vmax = base_vmax
                
                # if there is a boosted car in right lane
                if lane_idx == 1 and boosted_cars[lane_idx][pos]:
                    real_vmax = driver_types_dict[driver_type]['boosted_vmax']
                
                # Step 1: Acceleration
                if v < real_vmax:
                    v += 1
                
                # Step 2: Slowing down
                gap = get_gap_ahead_same_lane(lanes[lane_idx], pos)
                v = min(v, gap)
                
                # Step 3: Randomization
                if random_brake_override is not None:
                    random_brake_chance = random_brake_override
                else:
                    random_brake_chance = driver_types_dict[driver_type]['random_brake_chance']
                
                if v > 0 and np.random.random() < random_brake_chance:
                    v -= 1
                
                lanes[lane_idx][pos] = v
    
    return lanes

# move cars after get velocity
def move_cars(lanes, driver_types):
    length = len(lanes[0])
    new_lanes = [np.full(length, -1), np.full(length, -1)]
    new_driver_types = [np.full(length, -1), np.full(length, -1)]
    
    for lane_idx in range(2):
        for pos in reversed(range(length)):  
            if lanes[lane_idx][pos] >= 0:
                v = lanes[lane_idx][pos]
                new_pos = pos + v
                
                if new_pos < length:
                    if new_lanes[lane_idx][new_pos] == -1: 
                        new_lanes[lane_idx][new_pos] = v
                        new_driver_types[lane_idx][new_pos] = driver_types[lane_idx][pos]
                    else:
                        print('collusion')
                    
    return new_lanes, new_driver_types

# add car
def add_new_car_with_density_control(lanes, driver_types, target_density, driver_types_dict = None):
    if driver_types_dict is None:
        driver_types_dict = DRIVER_TYPES
    
    length = len(lanes[0])
    current_cars = sum(sum(lane >= 0) for lane in lanes)
    current_density = current_cars / (2 * length)

    if current_density < target_density:
        # instead of generate new cars in both lane, we choose a lane to add new car, higher possibility to generate in left lane(slow lane)
        generate_rate = [0.7, 0.3]
        lane_idx = np.random.choice([0, 1], p=generate_rate)
        
        forward_gap = get_gap_ahead_same_lane(lanes[lane_idx], 0)
        adj_forward_gap, _ = get_gaps_other_lane(lanes, lane_idx, 0)
        
        # if the start pos is empty
        if lanes[lane_idx][0] == -1:
            weights = [driver_type['spawn_weight'] for driver_type in driver_types_dict.values()]
            new_driver_type = np.random.choice(len(driver_types_dict), p=weights)
            driver_params = driver_types_dict[new_driver_type]
            
            # since it's on highway, we want the new car with a initial speed
            min_entry_speed = int(driver_params['vmax'] * 0.5)      # lower bound
            max_entry_speed = int(driver_params['vmax'])            # upper bound
            
            # to make sure it is safe when add new cars
            safe_entry_speed = min(max_entry_speed, forward_gap - 1, adj_forward_gap - 1)
            
            # add car when its initial speed greater than min_entry_speed, smaller than safe_entry_speed
            if safe_entry_speed >= min_entry_speed:
                initial_speed = np.random.randint(min_entry_speed, safe_entry_speed + 1, dtype=int)
                lanes[lane_idx][0] = initial_speed
                driver_types[lane_idx][0] = new_driver_type
    
    return lanes, driver_types

# code for update traffic for each step
def update_traffic_with_density(lanes, driver_types, target_density, random_brake_override = None, driver_types_dict=None):
    if driver_types_dict is None:
        driver_types_dict = DRIVER_TYPES
    
    # change lane
    lanes, driver_types, num_lane_change = change_lanes(lanes, driver_types, driver_types_dict)
    
    # update velocities
    lanes = update_velocities(lanes, driver_types, random_brake_override, driver_types_dict)
    
    # move cars
    lanes, driver_types = move_cars(lanes, driver_types)
    
    # add new car at start point
    lanes, driver_types = add_new_car_with_density_control(lanes, driver_types, target_density, driver_types_dict)
    
    return lanes, driver_types, num_lane_change

# code for calculate traffic measurements
def calculate_traffic_measurements(lanes_history, interval_size):
    actual_interval_size = min(len(lanes_history), interval_size)
    measure_interval = lanes_history[-actual_interval_size:]
    length = len(measure_interval[0][0])
    measure_point = length // 2
    
    total_flow = 0
    num_cars = 0
    total_velocity = 0
    
    current_lanes = lanes_history[-1]
    for lane_idx in range(2):
        car_positions = np.where(current_lanes[lane_idx] >= 0)[0]
        car_velocities = current_lanes[lane_idx][car_positions]
        
        num_cars += len(car_positions)
        total_velocity += np.sum(car_velocities)
        
        for lanes in measure_interval:
            flow = 0    # flow for each step
            for lane_idx in range(2):
                car_positions = np.where(lanes[lane_idx] >= 0)[0]
                for pos in car_positions:
                    velocity = lanes[lane_idx][pos]
                    if velocity > 0:
                        start_pos = pos - velocity
                        if start_pos <= measure_point < pos:
                            flow += 1
                            
            total_flow += flow
            
    avg_flow = total_flow / interval_size  if actual_interval_size > 0 else 0
    density = num_cars / (2 * length)
    avg_velocity = total_velocity / num_cars if num_cars > 0 else 0

    return avg_flow, density, avg_velocity


# code for placing obstacle
def place_obstacle(lanes, driver_types, position, lane, obstacle_type=3, driver_types_dict=None):
    if driver_types_dict is None:
        driver_types_dict = DRIVER_TYPES

    length = len(lanes[0])
    if position < 0 or position >= length:
        raise ValueError("position out of range")
    
    if lane not in [0, 1]:
        raise ValueError("incorrect lane index(have to be 0/1)")
    
    # check if there is a car at obstacle position, if yes, replace it by obstacle
    if lanes[lane][position] != -1:
        print(f"({lane}, {position}) already have a car, so relpace it by obstacle")

    vmax = driver_types_dict[obstacle_type]['vmax']
    lanes[lane][position] = vmax  
    driver_types[lane][position] = obstacle_type
    
    return lanes, driver_types

# calculate congestion measurement
def calculate_local_congestion(lanes, center_position, radius):
    length = len(lanes[0])
    
    start_pos = max(0, center_position - radius)
    end_pos = min(length, center_position + radius)
    
    num_vehicle = 0
    velocity_sum = 0
    
    for lane_idx in range(2):
        for pos in range(start_pos, end_pos):
            if lanes[lane_idx][pos] >= 0:
                num_vehicle += 1
                velocity_sum += lanes[lane_idx][pos]
                
    local_area_size = 2 * (end_pos - start_pos) 
    local_density = num_vehicle / local_area_size if local_area_size > 0 else 0
    local_velocity = velocity_sum / num_vehicle if num_vehicle > 0 else 0
    
    return local_density, local_velocity

# main code for simulation
def simulate_traffic_stability_with_obstacle(length, t0, steps, target_density, 
                                           obstacle_start_time, obstacle_position, obstacle_duration, obstacle_lane = 0, obstacle_type=3, measurement_radius=25, interval_size=10, random_brake_override=None, driver_types_dict=None):
    if driver_types_dict is None:
        driver_types_dict = DRIVER_TYPES
        
    lanes = [np.full(length, -1), np.full(length, -1)]  
    driver_types = [np.full(length, -1), np.full(length, -1)]  
    
    global_flows = []
    global_densities = []
    global_velocities = []
    lane_change_rates = []
    
    local_densities = []
    local_velocities = []

    
    lanes_history = []
    
    # first run the code t0 times, and begin the collection of data after the first t0 time steps
    for _ in range(t0):
        lanes, driver_types, _ =  update_traffic_with_density(lanes, driver_types, target_density, random_brake_override, driver_types_dict)
    
    obstacle_active = False
    obstacle_timer = 0
    
    for step in range(steps):
        lanes_history.append(copy.deepcopy(lanes))
        
        # place obstacle at given time at given position
        if step == obstacle_start_time:
            obstacle_active = True
            obstacle_timer = obstacle_duration if obstacle_duration > 0 else float('inf')
            
            lanes, driver_types = place_obstacle(
                lanes, driver_types, obstacle_position, obstacle_lane, 
                obstacle_type, driver_types_dict
            )
        
        lanes, driver_types, num_lane_change = update_traffic_with_density(lanes, driver_types, target_density, random_brake_override, driver_types_dict)
        
        if obstacle_active:
            # update the timer
            if obstacle_timer != float('inf'):
                obstacle_timer -= 1
                if obstacle_timer <= 0:
                    # remove it
                    obstacle_active = False
                    lanes[obstacle_lane][obstacle_position] = -1
                    driver_types[obstacle_lane][obstacle_position] = -1
        
        # global measurement
        if len(lanes_history) > interval_size:
            lanes_history.pop(0)
            

        flow, density, velocity = calculate_traffic_measurements(lanes_history, interval_size)
        global_flows.append(flow)
        global_densities.append(density)
        global_velocities.append(velocity)
        
        # local measurement around obstacle
        local_density, local_velocity= calculate_local_congestion(lanes, obstacle_position, measurement_radius)
        local_densities.append(local_density)
        local_velocities.append(local_velocity)
 

        current_vehicles = sum(sum(lane >= 0) for lane in lanes)
        lane_change_rate = num_lane_change / current_vehicles if current_vehicles > 0 else 0 
        lane_change_rates.append(lane_change_rate)
    
    return {
        "obstacle_start_time" : obstacle_start_time,
        "obstacle_end_time" : obstacle_start_time +  obstacle_duration,
        'global_flows': global_flows,
        'global_densities': global_densities,
        'global_velocities': global_velocities,
        'lane_change_rates': lane_change_rates,
        'local_densities': local_densities,
        'local_velocities': local_velocities
    }

# generate the plot
def plot_obstacle_impact(results):
    time_steps = range(len(results['global_densities']))
    fig, axs = plt.subplots(2, 1, figsize=(15, 6))

    # density
    axs[0].plot(time_steps, results['global_densities'], 'r-', label='Global Density')
    axs[0].plot(time_steps, results['local_densities'], 'b-', label='Local Density', alpha= 0.5)
    
    axs[0].axvline(x=results['obstacle_start_time'], color='k', linestyle='--', label='Obstacle Appear')
    axs[0].axvline(x=results['obstacle_end_time'], color='g', linestyle='--', label='Obstacle Disappear')
    
    axs[0].set_xlabel('time-steps')
    axs[0].set_ylabel('densities')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, alpha=0.3)
    
    # speed
    axs[1].plot(time_steps, results['global_velocities'], 'r-', label='Global Velocity')
    axs[1].plot(time_steps, results['local_velocities'], 'b-', label='Local Velocity',alpha= 0.5)
    
    axs[1].axvline(x=results['obstacle_start_time'], color='k', linestyle='--', label='Obstacle Appear')
    axs[1].axvline(x=results['obstacle_end_time'], color='g', linestyle='--', label='Obstacle Disappear')
    
    axs[1].set_xlabel('time-steps')
    axs[1].set_ylabel('velocities')
    axs[1].legend(loc='upper left')
    axs[1].grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    plt.savefig('obstacle_impact.png')

    
    return

def main():
    np.random.seed(20)

    print("start")
    
    length = 1000  
    t0 = 0      
    steps = 8000
    target_density = 1     
    obstacle_start_time = 3000
    obstacle_position = 500
    obstacle_duration = 1000


    results = simulate_traffic_stability_with_obstacle(length, t0, steps, target_density, 
                                            obstacle_start_time, obstacle_position, obstacle_duration, obstacle_lane = 0, obstacle_type=3, measurement_radius=10, interval_size=10, random_brake_override=None, driver_types_dict=None)
    
    plot_obstacle_impact(results)
    print("finished")

if __name__ == "__main__":
    main()