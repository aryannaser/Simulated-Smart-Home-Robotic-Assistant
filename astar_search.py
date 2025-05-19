import heapq

def manhattan_distance(p1, p2):
    """
    Calculate the Manhattan distance between two points.
    
    Args:
        p1: First point (x1, y1)
        p2: Second point (x2, y2)
        
    Returns:
        Manhattan distance |x1 - x2| + |y1 - y2|
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def astar_search(environment, start_pos, goal_pos):
    """
    Perform A* search to find a path from start_pos to goal_pos.
    
    Args:
        environment: Instance of HomeEnvironment
        start_pos: Starting position (x, y)
        goal_pos: Goal position (x, y)
        
    Returns:
        List of tuples representing the path [(x1, y1), (x2, y2), ...] or None if no path
    """
    # If start is the goal, return a single point path
    if start_pos == goal_pos:
        return [start_pos]
    
    # Initialize open set (priority queue: (f_score, position, path_so_far))
    open_set = [(manhattan_distance(start_pos, goal_pos), start_pos, [start_pos])]
    heapq.heapify(open_set)
    
    # Initialize closed set (positions we've already processed)
    closed_set = set()
    
    while open_set:
        # Get node with lowest f_score
        _, current_pos, path_so_far = heapq.heappop(open_set)
        
        # Check if we've reached the goal
        if current_pos == goal_pos:
            return path_so_far
        
        # Skip if we've already processed this node
        if current_pos in closed_set:
            continue
        
        # Add to closed set
        closed_set.add(current_pos)
        
        # Get valid neighbors
        neighbors = environment.get_valid_neighbors(current_pos[0], current_pos[1])
        
        # Process each neighbor
        for neighbor in neighbors:
            # Skip if already processed
            if neighbor in closed_set:
                continue
            
            # Calculate g_score (cost from start to neighbor)
            g_score = len(path_so_far)  # Each step costs 1
            
            # Calculate h_score (heuristic from neighbor to goal)
            h_score = manhattan_distance(neighbor, goal_pos)
            
            # Calculate f_score (total estimated cost)
            f_score = g_score + h_score
            
            # Create new path including this neighbor
            new_path = path_so_far + [neighbor]
            
            # Add to open set
            heapq.heappush(open_set, (f_score, neighbor, new_path))
    
    # If we get here, no path was found
    return None 