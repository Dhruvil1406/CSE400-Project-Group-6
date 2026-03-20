import os
import sys
import optparse
import xml.etree.ElementTree as ET
import random
import math
import time
import heapq

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

def get_random_edges():
    tree = ET.parse('map.net.xml')
    root = tree.getroot()
    edges_coords = {}
    for edge in root.findall('edge'):
        if 'function' not in edge.attrib:
            eid = edge.get('id')
            lanes = edge.findall('lane')
            if lanes:
                shape = lanes[0].get('shape')
                if shape:
                    coords = shape.split(' ')[0]
                    try:
                        x, y = map(float, coords.split(','))
                        edges_coords[eid] = (x, y)
                    except ValueError:
                        pass
    
    if len(edges_coords) < 3:
        return None
        
    all_edge_ids = list(edges_coords.keys())
    
    valid_pairs = []
    for _ in range(100):
        s = random.choice(all_edge_ids)
        e = random.choice(all_edge_ids)
        dist = math.hypot(edges_coords[s][0] - edges_coords[e][0], edges_coords[s][1] - edges_coords[e][1])
        valid_pairs.append((dist, s, e))
            
    # Sort pairs by distance descending to get the furthest 10%
    valid_pairs.sort(reverse=True, key=lambda x: x[0])
    top_pairs = valid_pairs[:10]
    
    # Randomly select a pair from the top 10%
    chosen = random.choice(top_pairs)
    start_edge = chosen[1]
    end_edge = chosen[2]
    
    return start_edge, end_edge

def visualize_astar_steps(start_edge_id, end_edge_id):
    print("\n--- A* Step-by-Step Evaluation ---\nParsing Network for Python A* Solver...")
    net_file = 'map.net.xml'
    # Read the network without internal edges to speed up calculation
    net = sumolib.net.readNet(net_file, withInternal=False)
    
    start_edge = net.getEdge(start_edge_id)
    end_edge = net.getEdge(end_edge_id)
    target_node = end_edge.getToNode()
    
    # Heuristic function: Straight line distance to destination (Euclidean)
    def heuristic(edge):
        node = edge.getToNode()
        return math.hypot(node.getCoord()[0] - target_node.getCoord()[0], 
                        node.getCoord()[1] - target_node.getCoord()[1])
                        
    open_set = []
    start_h = heuristic(start_edge)
    # heap stores tuples of (f_score, g_score, edge_id, path_so_far)
    heapq.heappush(open_set, (start_h, 0, start_edge.getID(), [start_edge.getID()]))
    g_scores = {start_edge.getID(): 0}
    step = 1
    
    while open_set:
        f_score, current_g, current_edge_id, path = heapq.heappop(open_set)
        current_edge = net.getEdge(current_edge_id)
        
        print(f"Step {step}: Evaluating Edge '{current_edge_id}' | g={current_g:.2f}, h={f_score-current_g:.2f}, f={f_score:.2f}")
        
        if current_edge_id == end_edge_id:
            print(f"Target '{end_edge_id}' reached in {step} steps!")
            break

        # If we have found a better way here since this was added to heap
        if current_g > g_scores.get(current_edge_id, float('inf')):
            continue

        outgoing = current_edge.getOutgoing()
        for next_edge in outgoing.keys():
            next_edge_id = next_edge.getID()
            cost = next_edge.getLength()
            tentative_g = current_g + cost
            
            # If we found a faster path to this neighbor
            if tentative_g < g_scores.get(next_edge_id, float('inf')):
                g_scores[next_edge_id] = tentative_g
                h = heuristic(next_edge)
                f = tentative_g + h
                new_path = list(path)
                new_path.append(next_edge_id)
                heapq.heappush(open_set, (f, tentative_g, next_edge_id, new_path))
                print(f"  -> Discovered neighbor '{next_edge_id}': g={tentative_g:.2f}, h={h:.2f}, f={f:.2f}")
                
        step += 1
    print("----------------------------------")

def run_astar_simulation(start_edge, end_edge):
    """
    Run deterministic A* simulation (headless) and find total travel time.
    """
    sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    # passing --routing-algorithm astar starts SUMO with A* as default routing
    sumoCmd = [sumoBinary, "-c", "route_optimization.sumocfg", "--no-step-log", "true", "--routing-algorithm", "astar"]
    
    # Pre-flight check: we need to ensure the start and end are even connected at all
    traci.start(sumoCmd)
    base_route = traci.simulation.findRoute(start_edge, end_edge)
    if not base_route.edges:
        print(f"FATAL: Start ({start_edge}) and End ({end_edge}) are fundamentally disconnected on this map.")
        traci.close()
        return False
        
    print(f"\n--- A* Shortest Path Calculation ---")
    print(f"Algorithm: A* Routing")
    print(f"Path Found: Yes")
    print(f"Total Edges Traversed: {len(base_route.edges)}")
    print(f"Total Distance (Shortest Path): {base_route.length:.2f} meters")
    print(f"Estimated Travel Time: {base_route.travelTime:.2f} seconds")
    print(f"------------------------------------\n")
    
    traci.close()
    
    # Showcase step-by-step A* evaluation before starting simulation
    visualize_astar_steps(start_edge, end_edge)
    
    print(f"--- Evaluating Deterministic A* Route ---")
    traci.start(sumoCmd)
    baseline_id = "astar_route"
    astar_time = float('inf')
    target_veh_id = "targetCar"
    
    start_time = time.time()
    
    try:
        traci.route.add(baseline_id, base_route.edges)
        traci.vehicle.add(target_veh_id, baseline_id, typeID="DEFAULT_VEHTYPE")
        traci.vehicle.setColor(target_veh_id, (0, 255, 0)) # Green for A*
        
        step = 0
        while step < 5000 and target_veh_id not in traci.simulation.getArrivedIDList():
            traci.simulationStep()
            step += 1
            if target_veh_id not in traci.vehicle.getIDList() and step > 10:
                break
                
        if target_veh_id in traci.simulation.getArrivedIDList():
            astar_time = step
            with open("astar_best_route.txt", "w") as f:
                f.write(",".join(base_route.edges))
                
            print(f"A* Route Travel Time: {astar_time} steps.")
        else:
            print("A* Route Failed to reach destination (timeout).")
            
    except traci.exceptions.TraCIException as e:
        print(f"A* Route creation failed: {e}")
        
    traci.close()
    
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"A* Computation Run Time (including sim): {computation_time:.4f} seconds")

    if astar_time != float('inf'):
        print("\n=== A* Deterministic Routing Results ===")
        print(f"Start Edge: {start_edge}")
        print(f"End Edge: {end_edge}")
        print(f"Total Simulation Travel Time: {astar_time} steps")
        return True
    return False

def run_showcase():
    """Run the GUI showcase with the best route."""
    try:
        with open("astar_best_route.txt", "r") as f:
            edges = f.read().split(",")
    except FileNotFoundError:
        print("No A* best route found. Cannot run showcase.")
        return

    print("Launching GUI to showcase A* route...")
    sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
    sumoCmd = [sumoBinary, "-c", "route_optimization.sumocfg", "--start"]

    traci.start(sumoCmd)
    
    route_id = "astar_optimal_route"
    traci.route.add(route_id, edges)
    target_veh_id = "targetCar"
    traci.vehicle.add(target_veh_id, route_id, typeID="DEFAULT_VEHTYPE")
    traci.vehicle.setColor(target_veh_id, (0, 255, 0)) # Green car for A*

    traci.gui.trackVehicle('View #0', target_veh_id)
    traci.gui.setZoom('View #0', 1000)

    try:
        step = 0
        while step < 10000:
            traci.simulationStep()
            if target_veh_id in traci.simulation.getArrivedIDList():
                print("Showcase complete. Target car arrived via A*.")
                break
            step += 1
    except traci.exceptions.FatalTraCIError:
        print("Showcase window closed by user.")
    finally:
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

if __name__ == "__main__":
    # Seed fixed for deterministic origin-destination selection
    random.seed(42)  # Ensures the same start and end edge is picked every time
    
    success = False
    attempts = 0
    max_attempts = 10
    
    while not success and attempts < max_attempts:
        attempts += 1
        print(f"\n--- Attempt {attempts} to find a valid Origin/Destination pair ---")
        result = get_random_edges()
        if not result:
            print("Failed to get edges from the map.")
            sys.exit(1)
            
        start_edge, end_edge = result
        
        print(f"Target Origin: {start_edge}")
        print(f"Target Destination: {end_edge}")
        
        success = run_astar_simulation(start_edge, end_edge)
        
    if success:
        run_showcase()
    else:
        print("\nCould not find a connected route across the map after multiple attempts.")
