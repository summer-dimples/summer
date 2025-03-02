import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

np.random.seed(32)


# dict to store different driver type(can also be used for reaction/...)
# make sure sapwn_weight sum up to 1
DRIVER_TYPES = {
    0: {  # Lorry
        'vmax': 3,
        'overtake_rate': 0,    
        'return_rate': 1,
        'spawn_weight': 0.1,
        'boost_chance': 0.3,    # this can be used to simulate the need time for boost?(for example, 0.5 means average 2 timestep before boost)
        'boosted_vmax': 5,
        'random_brake_chance' : 0.3      
    },
    1: {  # Regular Cars
        'vmax': 5,
        'overtake_rate': 0.5,    
        'return_rate': 0.5,     
        'spawn_weight': 0.7,
        'boost_chance': 0.3,   
        'boosted_vmax': 7,
        'random_brake_chance' : 0.3     
    },
    2: {  # Fast Cars
        'vmax': 7,
        'overtake_rate': 0.7,    
        'return_rate': 0,
        'spawn_weight':0.2,
        'boost_chance': 0.3,    
        'boosted_vmax': 9,
        'random_brake_chance' : 0.3        
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
def change_lanes(lanes, driver_types,  driver_types_dict=None):
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
    
    # fast lane
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
                
                boosted_vmax = base_vmax
                
                # if there is a boosted car in right lane
                if lane_idx == 1 and boosted_cars[lane_idx][pos]:
                    boosted_vmax = driver_types_dict[driver_type]['boosted_vmax']
                
                # Step 1: Acceleration
                if v < boosted_vmax:
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
    measure_interval = lanes_history[-interval_size:]
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
            
    avg_flow = total_flow / interval_size
    density = num_cars / (2 * length)  
    avg_velocity = total_velocity / num_cars if num_cars > 0 else 0

    return avg_flow, density, avg_velocity

# run the simulation
def simulate_traffic_with_density_control(length, t0, steps, target_density, interval_size = 10, random_brake_override=None, driver_types_dict=None):
    if driver_types_dict is None:
        driver_types_dict = DRIVER_TYPES
    
    lanes = [np.full(length, -1), np.full(length, -1)]  
    driver_types = [np.full(length, -1), np.full(length, -1)]  
    
    # first run the code t0 times, and begin the collection of data after the first t0 time steps
    for _ in range(t0):
        lanes, driver_types, num_lane_change = update_traffic_with_density(lanes, driver_types, target_density, random_brake_override)
    
    flows = []
    densities = []
    velocities = []
    lane_change_rates = []
    
    lanes_history = []
    driver_types_history = []
    total_vehicles = 0
    
    # simulate and start to record the data
    for step in range(steps):
        lanes_history.append(lanes)

        if len(lanes_history) > interval_size:
            lanes_history.pop(0)
            
        if len(lanes_history) == interval_size:
            flow, density, velocity = calculate_traffic_measurements(lanes_history, interval_size)
            flows.append(flow)
            densities.append(density)
            velocities.append(velocity)
            
        lanes, driver_types, num_lane_change = update_traffic_with_density(lanes, driver_types, target_density, random_brake_override)
        
        current_vehicles = sum(sum(lane >= 0) for lane in lanes)
        
        lane_change_rate = num_lane_change / current_vehicles if current_vehicles > 0 else 0 
        lane_change_rates.append(lane_change_rate)
        
    return flows, densities, velocities, lane_change_rates

def run_sensitivity_analysis():

    length = 1000   
    t0 = 1000       
    steps = 1500    
    base_target_density = 0.2  
    interval_size = 20 
    
    # Parameter ranges
    random_brake_params = np.linspace(0, 0.6, 11)  
    overtake_params = np.linspace(0, 1.0, 11)      
    vmax_offsets = [-2, -1, 0, 1, 2]             
    density_params = np.linspace(0.01, 0.11, 11)
    boost_chance_params = np.linspace(0.1, 0.9, 9)
    return_rate_params = np.linspace(0, 1.0, 11)
    
    # Storage for results
    random_brake_results = []
    overtake_results = []
    vmax_results = []
    density_results = []
    actual_densities = []
    boost_chance_results = []
    return_rate_results = []
    
    # effect of random braking probability
    for random_brake in random_brake_params:

        modified_driver_types = copy.deepcopy(DRIVER_TYPES)
        modified_driver_types['random_brake_chance'] = random_brake 
        
        flows, _, _, _ = simulate_traffic_with_density_control(
            length, t0, steps, base_target_density, interval_size, 
            driver_types_dict=modified_driver_types)
        
        avg_flow = np.mean(flows)
        random_brake_results.append(avg_flow)
        print(f"Random brake chance = {random_brake}, Average flow = {avg_flow:.4f}")
    
    # effect of overtake rate
    for overtake_rate in overtake_params:
        modified_driver_types = copy.deepcopy(DRIVER_TYPES)
        modified_driver_types['overtake_rate'] = overtake_rate  
        
        flows, _, _, _ = simulate_traffic_with_density_control(
            length, t0, steps, base_target_density, interval_size, 
            driver_types_dict=modified_driver_types)
        
        avg_flow = np.mean(flows)
        overtake_results.append(avg_flow)
        print(f"Overtake rate = {overtake_rate}, Average flow = {avg_flow:.4f}")
    
    # effect of maximum velocity
    for vmax_offset in vmax_offsets:
        modified_driver_types = copy.deepcopy(DRIVER_TYPES)
        
        for vehicle_type in modified_driver_types.values():
            original_vmax = vehicle_type['vmax']
            vehicle_type['vmax'] = original_vmax + vmax_offset
            vehicle_type['boosted_vmax'] = vehicle_type['boosted_vmax'] + vmax_offset
        
        flows, _, _, _ = simulate_traffic_with_density_control(
            length, t0, steps, base_target_density, interval_size, 
            driver_types_dict=modified_driver_types)
        
        avg_flow = np.mean(flows)
        vmax_results.append(avg_flow)
        print(f"Vmax offset = {vmax_offset}, Average flow = {avg_flow:.4f}")
    
    # effect of target density
    for density in density_params:
        flows, densities, _, _ = simulate_traffic_with_density_control(
            length, t0, steps, density, interval_size, 
            driver_types_dict=DRIVER_TYPES)
        
        avg_flow = np.mean(flows)
        actual_density = np.mean(densities)
        actual_densities.append(actual_density)
        density_results.append(avg_flow)
        print(f"Actual density = {actual_density}, Average flow = {avg_flow:.4f}")
    
    # effect of boost chance
    for boost_chance in boost_chance_params:
        modified_driver_types = copy.deepcopy(DRIVER_TYPES)
        for vehicle_type in modified_driver_types.values():
            vehicle_type['boost_chance'] = boost_chance
        
        flows, _, _, _ = simulate_traffic_with_density_control(
            length, t0, steps, base_target_density, interval_size, 
            driver_types_dict=modified_driver_types)
        
        avg_flow = np.mean(flows)
        boost_chance_results.append(avg_flow)
        print(f"Boost chance = {boost_chance}, Average flow = {avg_flow:.4f}")
    
    # effect of return rate
    for return_rate in return_rate_params:
        modified_driver_types = copy.deepcopy(DRIVER_TYPES)
        for vehicle_type in modified_driver_types.values():
            vehicle_type['return_rate'] = return_rate
        
        flows, _, _, _ = simulate_traffic_with_density_control(
            length, t0, steps, base_target_density, interval_size, 
            driver_types_dict=modified_driver_types)
        
        avg_flow = np.mean(flows)
        return_rate_results.append(avg_flow)
        print(f"Return rate = {return_rate}, Average flow = {avg_flow:.4f}")
    

    plt.figure(figsize=(24, 16))
    
    plt.subplot(2, 3, 1)
    plt.plot(random_brake_params, random_brake_results, 'o-', color='blue')
    plt.xlabel('Random Brake Probability')
    plt.ylabel('Average Flow')
    plt.title('Impact of Random Brake Probability on Flow')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(overtake_params, overtake_results, 'o-', color='green')
    plt.xlabel('Overtake Rate')
    plt.ylabel('Average Flow')
    plt.title('Impact of Overtake Rate on Flow')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(vmax_offsets, vmax_results, 'o-', color='red')
    plt.xlabel('Maximum Velocity')
    plt.ylabel('Average Flow')
    plt.title('Impact of Maximum Velocity on Flow')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(actual_densities, density_results, 'o-', color='purple')
    plt.xlabel('Density')
    plt.ylabel('Average Flow')
    plt.title('Impact of Density on Flow')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(boost_chance_params, boost_chance_results, 'o-', color='blue')
    plt.xlabel('Boost Chance')
    plt.ylabel('Average Flow')
    plt.title('Impact of Boost Chance on Flow')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(return_rate_params, return_rate_results, 'o-', color='green')
    plt.xlabel('Return Rate')
    plt.ylabel('Average Flow')
    plt.title('Impact of Return Rate on Flow')
    plt.grid(True)
    
    
    plt.tight_layout()
    plt.savefig('1.png')
    
    

    # Return results for further analysis
    return {
        'random_brake': (random_brake_params, random_brake_results),
        'overtake_rate': (overtake_params, overtake_results),
        'vmax_offset': (vmax_offsets, vmax_results),
        'density': (actual_densities, density_results),
        'boost_chance': (boost_chance_params, boost_chance_results),
        'return_rate': (return_rate_params, return_rate_results),
    }

def compare_brake_chances():
    length = 1000   
    t0 = 1000       
    steps = 1500     
    interval_size = 20 
    
    
    brake_chances = [0.0, 0.1, 0.3, 0.5]
    target_densities = np.linspace(0.01, 0.1, 15)
    all_results = {}
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    
    for i, brake_chance in enumerate(tqdm(brake_chances,desc="Simulating")):
        avg_velocities = []
        avg_densities = []
        avg_flows = []
        avg_lane_changes = []
        
        for target_density in target_densities:
            flows, densities, velocities, lane_change_rates = simulate_traffic_with_density_control(
                length=length,
                t0=t0,
                steps=steps,
                target_density=target_density,
                interval_size=interval_size,
                random_brake_override=brake_chance,
                driver_types_dict=None
            )
            
            avg_velocity = np.mean(velocities)
            avg_density = np.mean(densities)
            avg_flow = np.mean(flows)
            avg_lane_change = np.mean(lane_change_rates) if lane_change_rates else 0
            
            avg_velocities.append(avg_velocity)
            avg_densities.append(avg_density)
            avg_flows.append(avg_flow)
            avg_lane_changes.append(avg_lane_change)
              
        ax1.scatter(
            avg_densities, 
            avg_velocities, 
            color=colors[i], 
            marker=markers[i],
            label=f'probability of random brake = {brake_chance}'
        )
        
        z = np.polyfit(avg_densities, avg_velocities, 3)
        p = np.poly1d(z)
        x_range = np.linspace(min(avg_densities), max(avg_densities), 100)
        ax1.plot(x_range, p(x_range), color=colors[i], linestyle='--')

            
        ax2.scatter(
            avg_densities, 
            avg_flows, 
            color=colors[i], 
            marker=markers[i],
            label=f'probability of random brake = {brake_chance}'
        )

        z = np.polyfit(avg_densities, avg_flows, 3)
        p = np.poly1d(z)
        x_range = np.linspace(min(avg_densities), max(avg_densities), 100)
        ax2.plot(x_range, p(x_range), color=colors[i], linestyle='--')
        
        ax3.scatter(
            avg_densities, 
            avg_lane_changes, 
            color=colors[i], 
            marker=markers[i],
            label=f'probability of random brake = {brake_chance}'
        )
        

        z = np.polyfit(avg_densities, avg_lane_changes, 3)
        p = np.poly1d(z)
        x_range = np.linspace(min(avg_densities), max(avg_densities), 100)
        ax3.plot(x_range, p(x_range), color=colors[i], linestyle='--')

            
    ax1.set_xlabel('Density')
    ax1.set_ylabel('Speed')
    ax1.set_title('Speed-Density Relationship')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_xlabel('Density')
    ax2.set_ylabel('Flow')
    ax2.set_title('Flow-Density Relationship')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    ax3.set_xlabel('Density')
    ax3.set_ylabel('Lane Change Rate (changes per vehicle)')
    ax3.set_title('Lane Change Rate-Density Relationship')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('2.png')
    plt.show()
    
    return {
        'avg_densities': avg_densities,
        'avg_velocities': avg_velocities,
        'avg_flow': avg_flows,
        'avg_lane_changes': avg_lane_changes
    }


def main():

    print("start")
    
    results_1 = run_sensitivity_analysis()
    # comparing brake chances
    print("Comparing brake chances")
    results_2 = compare_brake_chances()
    
    print("finished")

if __name__ == "__main__":
    main()