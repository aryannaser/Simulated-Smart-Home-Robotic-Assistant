from astar_search import astar_search, manhattan_distance
from home_environment import HomeEnvironment
import numpy as np

def test_astar_search():
    # Create a simple grid environment for testing
    grid_layout = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Empty item locations for this test
    item_locations = {}
    
    # Create environment
    env = HomeEnvironment(grid_layout, item_locations)
    
    # Test 1: Path to same position
    print("Test 1: Path to same position")
    start_pos = (1, 1)
    goal_pos = (1, 1)
    path = astar_search(env, start_pos, goal_pos)
    print(f"Path from {start_pos} to {goal_pos}: {path}")
    
    assert path == [start_pos], "Path to same position should just be the position itself"
    print("✓ Same position test passed")
    print()
    
    # Test 2: Simple path
    print("Test 2: Simple path")
    start_pos = (1, 1)  # Top-left open space
    goal_pos = (5, 5)   # Bottom-right open space
    path = astar_search(env, start_pos, goal_pos)
    print(f"Path from {start_pos} to {goal_pos}: {path}")
    
    assert path is not None, "Should find a path"
    assert path[0] == start_pos, "Path should start at start_pos"
    assert path[-1] == goal_pos, "Path should end at goal_pos"
    assert_valid_path(path, env)
    print("✓ Simple path test passed")
    print()
    
    # Test 3: Path around obstacle
    print("Test 3: Path around obstacle")
    start_pos = (1, 3)  # Left side
    goal_pos = (5, 3)   # Right side (need to go around obstacle)
    path = astar_search(env, start_pos, goal_pos)
    print(f"Path from {start_pos} to {goal_pos}: {path}")
    
    assert path is not None, "Should find a path around obstacle"
    assert path[0] == start_pos, "Path should start at start_pos"
    assert path[-1] == goal_pos, "Path should end at goal_pos"
    assert_valid_path(path, env)
    print("✓ Path around obstacle test passed")
    print()
    
    # Test 4: Unreachable goal
    print("Test 4: Unreachable goal")
    # Create a grid with two separate areas that can't reach each other
    grid_layout_unreachable = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    env_unreachable = HomeEnvironment(grid_layout_unreachable, {})
    
    start_pos = (1, 1)  # Left side
    goal_pos = (5, 1)   # Right side, completely separated by a wall
    path = astar_search(env_unreachable, start_pos, goal_pos)
    print(f"Path from {start_pos} to {goal_pos} (unreachable): {path}")
    
    assert path is None, "Should not find a path to unreachable goal"
    print("✓ Unreachable goal test passed")
    
    print("\nAll A* search tests completed successfully!")

def assert_valid_path(path, env):
    """Helper function to check if a path is valid"""
    # Check that each step is a valid move (adjacent and not an obstacle)
    for i in range(len(path) - 1):
        curr_pos = path[i]
        next_pos = path[i + 1]
        
        # Check that the move is to an adjacent cell
        dx = abs(curr_pos[0] - next_pos[0])
        dy = abs(curr_pos[1] - next_pos[1])
        assert dx + dy == 1, f"Invalid move from {curr_pos} to {next_pos}"
        
        # Check that the destination is not an obstacle
        assert not env.is_obstacle(next_pos[0], next_pos[1]), f"Path includes obstacle at {next_pos}"

if __name__ == "__main__":
    test_astar_search() 