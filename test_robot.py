from robot import Robot
from home_environment import HomeEnvironment

def test_robot():
    # Sample grid layout
    grid_layout = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'kitchen', 'kitchen', 0, 0, 'living_room', 'living_room', 1],
        [1, 'kitchen', 'kitchen', 0, 0, 'living_room', 'living_room', 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 'bedroom', 'bedroom', 1, 1, 'bathroom', 'bathroom', 1],
        [1, 'bedroom', 'bedroom', 1, 1, 'bathroom', 'bathroom', 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Sample item locations
    item_locations = {
        'cup': (1, 1),        # kitchen
        'book': (5, 1),       # living room
        'phone': (1, 4),      # bedroom
        'toothbrush': (5, 4)  # bathroom
    }
    
    # Create HomeEnvironment instance
    env = HomeEnvironment(grid_layout, item_locations)
    
    # Test 1: Initialize robot and check initial state
    print("Test 1: Initialize robot")
    robot = Robot((3, 3))  # Start in the middle empty space
    print(f"Initial position: {robot.current_pos}")
    print(f"Initial item held: {robot.item_held}")
    
    assert robot.current_pos == (3, 3), "Robot should start at (3, 3)"
    assert robot.item_held is None, "Robot should not hold any item initially"
    print("✓ Robot initialization test passed")
    print()
    
    # Test 2: Test move_to
    print("Test 2: Test move_to")
    robot.move_to((2, 3))
    print(f"New position after move: {robot.current_pos}")
    
    assert robot.current_pos == (2, 3), "Robot should move to (2, 3)"
    print("✓ move_to test passed")
    print()
    
    # Test 3: Try to pickup item when not at item's location
    print("Test 3: Try pickup_item when not at item's location")
    success = robot.pickup_item('cup', env)
    print(f"Pickup success: {success}")
    print(f"Item held after failed pickup: {robot.item_held}")
    
    assert success == False, "Pickup should fail when not at item's location"
    assert robot.item_held is None, "Robot should still not hold any item"
    assert env.get_item_location('cup') == (1, 1), "Cup should remain at its original location"
    print("✓ Pickup not at location test passed")
    print()
    
    # Test 4: Move to item's location and pickup
    print("Test 4: Move to item's location and pickup")
    robot.move_to((1, 1))  # Move to cup's location
    print(f"New position (at cup): {robot.current_pos}")
    success = robot.pickup_item('cup', env)
    print(f"Pickup success: {success}")
    print(f"Item held after pickup: {robot.item_held}")
    print(f"Cup's location after pickup: {env.get_item_location('cup')}")
    
    assert success == True, "Pickup should succeed at item's location"
    assert robot.item_held == 'cup', "Robot should be holding the cup"
    assert env.get_item_location('cup') is None, "Cup should be held (location is None)"
    print("✓ Pickup at location test passed")
    print()
    
    # Test 5: Try to putdown item
    print("Test 5: Putdown item")
    robot.move_to((3, 3))  # Move back to the middle
    print(f"New position for putdown: {robot.current_pos}")
    success = robot.putdown_item(env)
    print(f"Putdown success: {success}")
    print(f"Item held after putdown: {robot.item_held}")
    print(f"Cup's location after putdown: {env.get_item_location('cup')}")
    
    assert success == True, "Putdown should succeed when holding an item"
    assert robot.item_held is None, "Robot should not hold any item after putdown"
    assert env.get_item_location('cup') == (3, 3), "Cup should be at robot's location after putdown"
    print("✓ Putdown test passed")
    print()
    
    # Test 6: Try to putdown when not holding anything
    print("Test 6: Putdown when not holding anything")
    success = robot.putdown_item(env)
    print(f"Putdown success: {success}")
    
    assert success == False, "Putdown should fail when not holding anything"
    print("✓ Putdown when not holding test passed")
    
    print("\nAll robot tests completed successfully!")

if __name__ == "__main__":
    test_robot() 