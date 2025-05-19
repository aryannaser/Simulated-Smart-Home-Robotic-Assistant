from robot import Robot
from home_environment import HomeEnvironment
import random
import numpy as np

def test_robot_with_hmm():
    # Create a simple grid environment for testing
    grid_layout = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 'kitchen', 'kitchen', 0, 0, 'living_room', 1],
        [1, 'kitchen', 'kitchen', 0, 0, 'living_room', 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 'bedroom', 'bedroom', 0, 0, 'bathroom', 1],
        [1, 'bedroom', 'bedroom', 0, 0, 'bathroom', 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Sample item locations
    item_locations = {
        'cup': (1, 1),       # kitchen
        'book': (5, 1),      # living room
        'phone': (1, 4),     # bedroom
        'toothbrush': (5, 4) # bathroom
    }
    
    # Create HomeEnvironment instance
    env = HomeEnvironment(grid_layout, item_locations)
    
    # Generate all possible locations
    all_possible_locations = []
    for y in range(env.height):
        for x in range(env.width):
            if not env.is_obstacle(x, y):
                all_possible_locations.append((x, y))
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define room observations
    room_observations = [
        'kitchen_sensed',
        'living_room_sensed',
        'bedroom_sensed',
        'bathroom_sensed',
        'unknown_sensed',
        'action_succeeded',
        'action_failed'
    ]
    
    # Test 1: Initialize robot with uniform belief
    print("Test 1: Initialize robot with uniform belief")
    robot = Robot(None, all_possible_locations, room_observations, env)
    
    most_likely_pos = robot.get_most_likely_pos()
    print(f"Initial most likely position: {most_likely_pos}")
    print(f"Initial item held: {robot.item_held}")
    
    # All positions should have equal probability initially
    belief_values = list(robot.hmm.belief_state.values())
    assert all(abs(p - belief_values[0]) < 1e-10 for p in belief_values), "All initial belief probabilities should be equal"
    assert robot.item_held is None, "Robot should not be holding any item initially"
    print("✓ Robot initialization test passed")
    print()
    
    # Test 2: Initialize robot with certain belief
    print("Test 2: Initialize robot with certain belief")
    certain_loc = (2, 1)  # Kitchen
    certain_belief = {loc: 1.0 if loc == certain_loc else 0.0 for loc in all_possible_locations}
    robot_certain = Robot(certain_belief, all_possible_locations, room_observations, env)
    
    most_likely_pos = robot_certain.get_most_likely_pos()
    print(f"Most likely position with certain belief: {most_likely_pos}")
    
    assert most_likely_pos == certain_loc, f"Most likely position should be {certain_loc}"
    print("✓ Robot with certain belief test passed")
    print()
    
    # Test 3: Move and observe
    print("Test 3: Move and observe belief updates")
    robot = Robot(certain_belief, all_possible_locations, room_observations, env)
    print(f"Initial most likely position: {robot.get_most_likely_pos()}")
    
    # Move right (towards living room)
    print("Moving right (1, 0)")
    robot.move_to((1, 0))
    
    new_pos = robot.get_most_likely_pos()
    print(f"New most likely position: {new_pos}")
    
    # Get top 3 belief positions
    sorted_beliefs = sorted(robot.hmm.belief_state.items(), key=lambda x: x[1], reverse=True)
    print("Top 3 belief positions:")
    for pos, prob in sorted_beliefs[:3]:
        print(f"{pos}: {prob:.6f}")
    
    # The belief should have shifted towards the right (but still uncertain)
    assert new_pos[0] >= certain_loc[0], "Position should have moved to the right"
    print("✓ Movement test passed")
    print()
    
    # Test 4: Pickup item
    print("Test 4: Pickup item")
    # Create robot with certain belief at cup location
    cup_loc = env.get_item_location('cup')
    cup_belief = {loc: 1.0 if loc == cup_loc else 0.0 for loc in all_possible_locations}
    robot = Robot(cup_belief, all_possible_locations, room_observations, env)
    
    print(f"Robot certain location (cup): {robot.get_most_likely_pos()}")
    print(f"Cup location: {env.get_item_location('cup')}")
    
    # Try to pickup cup
    success = robot.pickup_item('cup', env)
    print(f"Pickup success: {success}")
    print(f"Item held after pickup: {robot.item_held}")
    print(f"Cup location after pickup: {env.get_item_location('cup')}")
    
    assert success == True, "Pickup should succeed at cup's location"
    assert robot.item_held == 'cup', "Robot should be holding the cup"
    assert env.get_item_location('cup') is None, "Cup should be held (location is None)"
    print("✓ Pickup test passed")
    print()
    
    # Test 5: Move and putdown item
    print("Test 5: Move and putdown item")
    # Move to living room
    for _ in range(3):
        robot.move_to((1, 0))
    
    living_room_pos = robot.get_most_likely_pos()
    print(f"New position after movement: {living_room_pos}")
    
    # Put down cup
    success = robot.putdown_item(env)
    print(f"Putdown success: {success}")
    print(f"Item held after putdown: {robot.item_held}")
    print(f"Cup location after putdown: {env.get_item_location('cup')}")
    
    assert success == True, "Putdown should succeed when holding an item"
    assert robot.item_held is None, "Robot should not hold any item after putdown"
    assert env.get_item_location('cup') == living_room_pos, "Cup should be at robot's location after putdown"
    print("✓ Move and putdown test passed")
    
    print("\nAll robot with HMM tests completed successfully!")

if __name__ == "__main__":
    test_robot_with_hmm() 