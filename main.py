import numpy as np
import random
from home_environment import HomeEnvironment
from robot import Robot
from action_schema import ActionSchema
from planner import forward_planner

def create_environment():
    """Create a sample home environment."""
    # Create a grid layout with rooms
    grid_layout = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'kitchen', 'kitchen', 'kitchen', 1, 'living_room', 'living_room', 'living_room', 'living_room', 1],
        [1, 'kitchen', 'kitchen', 'kitchen', 1, 'living_room', 'living_room', 'living_room', 'living_room', 1],
        [1, 'kitchen', 'kitchen', 'kitchen', 0, 0, 0, 'living_room', 'living_room', 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 'bedroom', 0, 0, 0, 0, 0, 0, 'bathroom', 1],
        [1, 'bedroom', 'bedroom', 'bedroom', 1, 'bathroom', 'bathroom', 'bathroom', 'bathroom', 1],
        [1, 'bedroom', 'bedroom', 'bedroom', 1, 'bathroom', 'bathroom', 'bathroom', 'bathroom', 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    # Define item locations
    item_locations = {
        'cup': (1, 1),        # kitchen
        'book': (6, 1),       # living room
        'phone': (1, 6),      # bedroom
        'toothbrush': (6, 6)  # bathroom
    }
    
    # Create the environment
    return HomeEnvironment(grid_layout, item_locations)

def create_action_schemas():
    """Create action schemas for planning."""
    # GoTo action schema
    goto_action = ActionSchema(
        name="GoTo",
        parameters=('room',),
        preconditions={
            ('At', 'robot', 'current_room'),
            ('Connected', 'current_room', 'room')
        },
        add_effects={
            ('At', 'robot', 'room')
        },
        delete_effects={
            ('At', 'robot', 'current_room')
        }
    )
    
    # PickUp action schema
    pickup_action = ActionSchema(
        name="PickUp",
        parameters=('item', 'room'),
        preconditions={
            ('At', 'robot', 'room'),
            ('At', 'item', 'room'),
            ('Holding', 'robot', 'nothing')
        },
        add_effects={
            ('Holding', 'robot', 'item')
        },
        delete_effects={
            ('At', 'item', 'room'),
            ('Holding', 'robot', 'nothing')
        }
    )
    
    # PutDown action schema
    putdown_action = ActionSchema(
        name="PutDown",
        parameters=('item', 'room'),
        preconditions={
            ('At', 'robot', 'room'),
            ('Holding', 'robot', 'item')
        },
        add_effects={
            ('At', 'item', 'room'),
            ('Holding', 'robot', 'nothing')
        },
        delete_effects={
            ('Holding', 'robot', 'item')
        }
    )
    
    return [goto_action, pickup_action, putdown_action]

def parse_user_goal(user_input, available_items, available_rooms):
    """
    Parse a user's natural language goal into planner goal predicates.
    
    Args:
        user_input: String containing the user's command
        available_items: List of available items
        available_rooms: List of available rooms
        
    Returns:
        Set of goal predicates or None if parsing failed
    """
    user_input = user_input.lower()
    goal_preds = set()
    
    # Fetch item command (e.g., "fetch cup")
    for item in available_items:
        if f"fetch {item}" in user_input or f"bring {item}" in user_input or f"get {item}" in user_input:
            # Default to bringing item to living room if no delivery location specified
            delivery_room = 'living_room'
            
            # Check if a specific destination is mentioned
            for room in available_rooms:
                if f"to {room}" in user_input or f"to the {room}" in user_input:
                    delivery_room = room
                    break
            
            # Goal: robot holding the item and at the delivery room
            goal_preds.add(('Holding', 'robot', item))
            goal_preds.add(('At', 'robot', delivery_room))
            return goal_preds
    
    # Go to room command (e.g., "go to kitchen")
    for room in available_rooms:
        if f"go to {room}" in user_input or f"move to {room}" in user_input or f"navigate to {room}" in user_input:
            goal_preds.add(('At', 'robot', room))
            return goal_preds
    
    # Put down item command (e.g., "put down cup")
    for item in available_items:
        if f"put down {item}" in user_input or f"drop {item}" in user_input or f"place {item}" in user_input:
            # Need to find where to put the item
            delivery_room = None
            
            for room in available_rooms:
                if f"in {room}" in user_input or f"in the {room}" in user_input:
                    delivery_room = room
                    break
            
            # If no room specified, assume current room
            if not delivery_room:
                return None  # Need the current room from the robot's state
            
            # Goal: item at delivery room and robot holding nothing
            goal_preds.add(('At', 'item', delivery_room))
            goal_preds.add(('Holding', 'robot', 'nothing'))
            return goal_preds
    
    # Failed to parse
    return None

def display_belief_distribution(belief_state, environment, top_n=5):
    """
    Display the top N most likely positions from the belief state.
    
    Args:
        belief_state: Dictionary mapping positions to probabilities
        environment: The HomeEnvironment instance
        top_n: Number of top positions to display
    """
    sorted_beliefs = sorted(belief_state.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} belief positions:")
    for i, (pos, prob) in enumerate(sorted_beliefs[:top_n]):
        pos_type = "Unknown"
        x, y = pos
        room_type = environment.get_room_type(x, y)
        if room_type:
            pos_type = room_type
        
        print(f"{i+1}. Position {pos} ({pos_type}): {prob:.4f}")

def main():
    """Main simulation loop."""
    # Create the environment
    environment = create_environment()
    
    # Generate all possible locations
    all_possible_locations = []
    for y in range(environment.height):
        for x in range(environment.width):
            if not environment.is_obstacle(x, y):
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
    
    # Create robot with uniform belief
    robot = Robot(None, all_possible_locations, room_observations, environment)
    
    # Define action schemas
    action_schemas = create_action_schemas()
    
    # Available items and rooms for goal parsing
    available_items = ['cup', 'book', 'phone', 'toothbrush']
    available_rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
    
    print("Welcome to the Smart Home Robot Assistant Simulation!")
    print("The robot uses probabilistic localization (HMM) and planning to execute tasks.")
    print("Available commands:")
    print("  - 'go to <room>' - Navigate to a specific room")
    print("  - 'fetch <item>' - Fetch an item (will bring to living room by default)")
    print("  - 'fetch <item> to <room>' - Fetch an item to a specific room")
    print("  - 'quit' - Exit the simulation")
    print()
    
    # Main loop
    running = True
    while running:
        # Get current robot state
        most_likely_pos = robot.get_most_likely_pos()
        room_type = environment.get_room_type(most_likely_pos[0], most_likely_pos[1])
        
        print(f"\nRobot believes it is at: {most_likely_pos} (Room: {room_type})")
        print(f"Robot is holding: {robot.item_held if robot.item_held else 'nothing'}")
        
        # Display top belief positions
        display_belief_distribution(robot.hmm.belief_state, environment, top_n=3)
        
        # Get user input
        user_input = input("\nEnter command (or 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            running = False
            continue
        
        # Get current state for planning
        current_state_preds = robot.current_world_state_for_planner(environment)
        
        # Parse user input into goal predicates
        goal_preds = parse_user_goal(user_input, available_items, available_rooms)
        
        if not goal_preds:
            print("Sorry, I didn't understand that command. Please try again.")
            continue
        
        print(f"Understood goal: {goal_preds}")
        
        # Generate plan
        plan = forward_planner(current_state_preds, goal_preds, action_schemas)
        
        if plan:
            print(f"Plan found: {plan}")
            
            # Execute plan
            success = robot.execute_plan(plan, environment)
            
            if success:
                print(f"Plan executed successfully!")
                print(f"Robot now believes it is at: {robot.get_most_likely_pos()}")
                print(f"Robot is now holding: {robot.item_held if robot.item_held else 'nothing'}")
            else:
                print("Plan execution failed.")
        else:
            print("No plan found for the given goal.")
    
    print("Simulation ended. Goodbye!")

if __name__ == "__main__":
    main() 