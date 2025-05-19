from home_environment import HomeEnvironment
import numpy as np

def test_home_environment():
    # Sample grid layout
    # 0: empty, 1: obstacle, string: room type
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
        'cup': (1, 1),       # kitchen
        'book': (5, 1),      # living room
        'phone': (1, 4),     # bedroom
        'toothbrush': (5, 4) # bathroom
    }
    
    # Create HomeEnvironment instance
    env = HomeEnvironment(grid_layout, item_locations)
    
    # Test is_obstacle method
    print("Testing is_obstacle:")
    print(f"Obstacle (0, 0): {env.is_obstacle(0, 0)}")  # Should be True (wall)
    print(f"Empty cell (3, 3): {env.is_obstacle(3, 3)}")  # Should be False
    print(f"Room cell (1, 1): {env.is_obstacle(1, 1)}")  # Should be False (kitchen)
    print(f"Out of bounds (-1, 0): {env.is_obstacle(-1, 0)}")  # Should be True
    
    # Add assertions for is_obstacle
    assert env.is_obstacle(0, 0) == True, "Wall should be an obstacle"
    assert env.is_obstacle(3, 3) == False, "Empty cell should not be an obstacle"
    assert env.is_obstacle(1, 1) == False, "Room cell should not be an obstacle"
    assert env.is_obstacle(-1, 0) == True, "Out of bounds should be an obstacle"
    print("✓ is_obstacle tests passed")
    print()
    
    # Test get_room_type method
    print("Testing get_room_type:")
    print(f"Kitchen cell (1, 1): {env.get_room_type(1, 1)}")  # Should be 'kitchen'
    print(f"Empty cell (3, 3): {env.get_room_type(3, 3)}")  # Should be None
    print(f"Obstacle (0, 0): {env.get_room_type(0, 0)}")  # Should be None
    
    # Add assertions for get_room_type
    assert env.get_room_type(1, 1) == 'kitchen', "Should return 'kitchen'"
    assert env.get_room_type(3, 3) == None, "Empty cell should return None"
    assert env.get_room_type(0, 0) == None, "Obstacle should return None"
    print("✓ get_room_type tests passed")
    print()
    
    # Test get_item_location method
    print("Testing get_item_location:")
    print(f"Cup location: {env.get_item_location('cup')}")  # Should be (1, 1)
    print(f"Non-existent item: {env.get_item_location('laptop')}")  # Should be None
    
    # Add assertions for get_item_location
    assert env.get_item_location('cup') == (1, 1), "Cup should be at (1, 1)"
    assert env.get_item_location('laptop') == None, "Non-existent item should return None"
    print("✓ get_item_location tests passed")
    print()
    
    # Test update_item_location method
    print("Testing update_item_location:")
    print(f"Cup location before update: {env.get_item_location('cup')}")
    env.update_item_location('cup', (3, 3))
    print(f"Cup location after update: {env.get_item_location('cup')}")
    
    # Add assertion for update_item_location
    assert env.get_item_location('cup') == (3, 3), "Cup should be updated to (3, 3)"
    
    env.update_item_location('cup', None)  # Cup is held
    print(f"Cup location when held: {env.get_item_location('cup')}")
    
    # Add assertion for held item
    assert env.get_item_location('cup') == None, "Cup should be None when held"
    print("✓ update_item_location tests passed")
    print()
    
    # Test get_valid_neighbors method
    print("Testing get_valid_neighbors:")
    # Middle cell (3, 3) should have 4 neighbors
    neighbors_3_3 = env.get_valid_neighbors(3, 3)
    print(f"Neighbors of (3, 3): {neighbors_3_3}")
    
    # Corner room cell (1, 1) should have fewer neighbors due to walls
    neighbors_1_1 = env.get_valid_neighbors(1, 1)
    print(f"Neighbors of (1, 1): {neighbors_1_1}")
    
    # Cell next to obstacle
    neighbors_2_4 = env.get_valid_neighbors(2, 4)
    print(f"Neighbors of (2, 4): {neighbors_2_4}")
    
    # Add assertions for get_valid_neighbors
    assert len(neighbors_3_3) == 3, "Middle cell should have 3 neighbors"
    assert (2, 1) in neighbors_1_1, "Right neighbor should be in neighbors of (1, 1)"
    assert (1, 2) in neighbors_1_1, "Bottom neighbor should be in neighbors of (1, 1)"
    assert len(neighbors_2_4) <= 4, "Cell next to obstacle should have at most 4 neighbors"
    print("✓ get_valid_neighbors tests passed")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_home_environment() 