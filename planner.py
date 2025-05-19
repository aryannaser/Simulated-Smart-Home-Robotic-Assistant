from collections import deque
from action_schema import ActionSchema
import copy

def is_applicable(action_schema, state, parameter_bindings):
    """
    Check if an action is applicable in the current state.
    
    Args:
        action_schema: The ActionSchema object
        state: Set of predicates representing the current world state
        parameter_bindings: Dictionary mapping parameter names to values
        
    Returns:
        True if the action is applicable, False otherwise
    """
    # Special case for GoTo action
    if action_schema.name == "GoTo":
        room_param = parameter_bindings.get('room')
        # Find where the robot is currently
        current_room = None
        for state_pred in state:
            if state_pred[0] == "At" and state_pred[1] == "robot":
                current_room = state_pred[2]
                break
        
        if current_room is None:
            return False
        
        # Check if there's a connection between current_room and target room
        connection_pred = ('Connected', current_room, room_param)
        if connection_pred not in state:
            return False
        
        # Check that we're not already at the target room
        if current_room == room_param:
            return False
        
        return True
    
    # Special case for PickUp action
    elif action_schema.name == "PickUp":
        item_param = parameter_bindings.get('item')
        room_param = parameter_bindings.get('room')
        
        # Check if robot is at the room
        robot_at_room = ('At', 'robot', room_param) in state
        
        # Check if item is at the room
        item_at_room = ('At', item_param, room_param) in state
        
        # Check if robot is holding nothing
        holding_nothing = ('Holding', 'robot', 'nothing') in state
        
        return robot_at_room and item_at_room and holding_nothing
    
    # Special case for PutDown action
    elif action_schema.name == "PutDown":
        item_param = parameter_bindings.get('item')
        room_param = parameter_bindings.get('room')
        
        # Check if robot is at the room
        robot_at_room = ('At', 'robot', room_param) in state
        
        # Check if robot is holding the item
        holding_item = ('Holding', 'robot', item_param) in state
        
        return robot_at_room and holding_item
    
    # Generic approach for other actions
    else:
        # Apply parameter bindings to preconditions
        ground_preconditions = []
        for pred in action_schema.preconditions:
            ground_pred = tuple(parameter_bindings.get(param, param) if param in parameter_bindings else param for param in pred)
            ground_preconditions.append(ground_pred)
        
        # Check if all ground preconditions are in the state
        return all(pred in state for pred in ground_preconditions)

def apply_action(action_schema, state, parameter_bindings):
    """
    Apply an action to a state and return the resulting state.
    
    Args:
        action_schema: The ActionSchema object
        state: Set of predicates representing the current world state
        parameter_bindings: Dictionary mapping parameter names to values
        
    Returns:
        New state after applying the action
    """
    # Create a mutable copy of the state
    new_state = set(state)
    
    # Special case for GoTo action
    if action_schema.name == "GoTo":
        room_param = parameter_bindings.get('room')
        
        # Find where the robot is currently
        for state_pred in state:
            if state_pred[0] == "At" and state_pred[1] == "robot":
                current_room = state_pred[2]
                
                # Remove current location
                new_state.remove(('At', 'robot', current_room))
                
                # Add new location
                new_state.add(('At', 'robot', room_param))
                
                break
    
    # Special case for PickUp action
    elif action_schema.name == "PickUp":
        item_param = parameter_bindings.get('item')
        room_param = parameter_bindings.get('room')
        
        # Remove item from room
        new_state.remove(('At', item_param, room_param))
        
        # Remove holding nothing
        new_state.remove(('Holding', 'robot', 'nothing'))
        
        # Add holding item
        new_state.add(('Holding', 'robot', item_param))
    
    # Special case for PutDown action
    elif action_schema.name == "PutDown":
        item_param = parameter_bindings.get('item')
        room_param = parameter_bindings.get('room')
        
        # Remove holding item
        new_state.remove(('Holding', 'robot', item_param))
        
        # Add item at room
        new_state.add(('At', item_param, room_param))
        
        # Add holding nothing
        new_state.add(('Holding', 'robot', 'nothing'))
    
    # Generic approach for other actions
    else:
        # Apply parameter bindings to delete effects and remove them from the state
        for pred in action_schema.delete_effects:
            ground_pred = tuple(parameter_bindings.get(param, param) if param in parameter_bindings else param for param in pred)
            if ground_pred in new_state:
                new_state.remove(ground_pred)
        
        # Apply parameter bindings to add effects and add them to the state
        for pred in action_schema.add_effects:
            ground_pred = tuple(parameter_bindings.get(param, param) if param in parameter_bindings else param for param in pred)
            new_state.add(ground_pred)
    
    return new_state

def find_possible_parameter_bindings(action_schema, state, objects):
    """
    Find possible parameter bindings for an action schema.
    
    Args:
        action_schema: The ActionSchema object
        state: Set of predicates representing the current world state
        objects: Set of available objects in the world
        
    Returns:
        List of dictionaries, each mapping parameter names to values
    """
    # Extract parameters from action schema
    parameters = action_schema.parameters
    
    # For each parameter, find possible values
    possible_values = {}
    for param in parameters:
        # For simplicity, assume all objects can be bound to any parameter
        possible_values[param] = objects
    
    # Generate all possible parameter binding combinations
    # This is a simplified approach using a recursive helper function
    def generate_bindings(params, current_binding=None):
        if current_binding is None:
            current_binding = {}
        
        if not params:
            return [current_binding]
        
        param = params[0]
        remaining_params = params[1:]
        bindings = []
        
        for value in possible_values[param]:
            new_binding = current_binding.copy()
            new_binding[param] = value
            bindings.extend(generate_bindings(remaining_params, new_binding))
        
        return bindings
    
    return generate_bindings(list(parameters))

def forward_planner(current_state_preds, goal_preds, action_schemas, max_depth=10):
    """
    Plan using forward search.
    
    Args:
        current_state_preds: Set of predicate tuples representing the current world state
        goal_preds: Set of predicate tuples that must all be true in the goal state
        action_schemas: List of ActionSchema instances
        max_depth: Maximum plan length to consider
        
    Returns:
        List of instantiated action tuples [(action_name, param1, ...), ...] or None if no plan found
    """
    # Convert predicate sets to frozensets for hashability
    initial_state = frozenset(current_state_preds)
    goal = frozenset(goal_preds)
    
    # Check if initial state already satisfies goal
    if goal.issubset(initial_state):
        return []  # Empty plan, goal already satisfied
    
    # Extract all objects from predicates for parameter binding
    objects = set()
    for pred in current_state_preds:
        for arg in pred[1:]:  # Skip predicate name
            if isinstance(arg, str):
                objects.add(arg)
    
    # Queue for BFS: (state, plan_so_far)
    queue = deque([(initial_state, [])])
    
    # Set to track visited states
    visited_states = {initial_state}
    
    while queue and len(queue[0][1]) < max_depth:
        state, plan = queue.popleft()
        
        # Try each action schema
        for action_schema in action_schemas:
            # For GoTo action, we need to handle the room parameter specially
            if action_schema.name == "GoTo":
                # Find all rooms in the current state
                rooms = set()
                connections = {}
                
                current_room = None
                for pred in state:
                    if pred[0] == "At" and pred[1] == "robot":
                        current_room = pred[2]
                    elif pred[0] == "Connected":
                        from_room = pred[1]
                        to_room = pred[2]
                        if from_room not in connections:
                            connections[from_room] = []
                        connections[from_room].append(to_room)
                        rooms.add(to_room)
                
                # Try to go to each connected room
                for target_room in rooms:
                    if current_room in connections and target_room in connections[current_room]:
                        param_bindings = {"room": target_room}
                        
                        if is_applicable(action_schema, state, param_bindings):
                            # Apply action
                            new_state = apply_action(action_schema, state, param_bindings)
                            new_state_frozen = frozenset(new_state)  # Make hashable
                            
                            # Build instantiated action
                            instantiated_action = (action_schema.name, target_room)
                            
                            # Build new plan
                            new_plan = plan + [instantiated_action]
                            
                            # Check if goal is reached
                            if goal.issubset(new_state_frozen):
                                return new_plan
                            
                            # Add to queue if not visited
                            if new_state_frozen not in visited_states:
                                visited_states.add(new_state_frozen)
                                queue.append((new_state_frozen, new_plan))
            
            # For PickUp action
            elif action_schema.name == "PickUp":
                # Find all items and rooms in the current state
                items_in_rooms = {}
                robot_room = None
                holding_nothing = False
                
                for pred in state:
                    if pred[0] == "At" and pred[1] == "robot":
                        robot_room = pred[2]
                    elif pred[0] == "At" and pred[1] != "robot":
                        item = pred[1]
                        room = pred[2]
                        items_in_rooms[(item, room)] = True
                    elif pred[0] == "Holding" and pred[1] == "robot" and pred[2] == "nothing":
                        holding_nothing = True
                
                # Try to pick up each item in the current room
                if robot_room and holding_nothing:
                    for (item, room) in items_in_rooms:
                        if room == robot_room:
                            param_bindings = {"item": item, "room": room}
                            
                            if is_applicable(action_schema, state, param_bindings):
                                # Apply action
                                new_state = apply_action(action_schema, state, param_bindings)
                                new_state_frozen = frozenset(new_state)  # Make hashable
                                
                                # Build instantiated action
                                instantiated_action = (action_schema.name, item, room)
                                
                                # Build new plan
                                new_plan = plan + [instantiated_action]
                                
                                # Check if goal is reached
                                if goal.issubset(new_state_frozen):
                                    return new_plan
                                
                                # Add to queue if not visited
                                if new_state_frozen not in visited_states:
                                    visited_states.add(new_state_frozen)
                                    queue.append((new_state_frozen, new_plan))
            
            # For PutDown action
            elif action_schema.name == "PutDown":
                # Find what the robot is holding and which room it's in
                holding_item = None
                robot_room = None
                
                for pred in state:
                    if pred[0] == "Holding" and pred[1] == "robot" and pred[2] != "nothing":
                        holding_item = pred[2]
                    elif pred[0] == "At" and pred[1] == "robot":
                        robot_room = pred[2]
                
                # Try to put down the item in the current room
                if holding_item and robot_room:
                    param_bindings = {"item": holding_item, "room": robot_room}
                    
                    if is_applicable(action_schema, state, param_bindings):
                        # Apply action
                        new_state = apply_action(action_schema, state, param_bindings)
                        new_state_frozen = frozenset(new_state)  # Make hashable
                        
                        # Build instantiated action
                        instantiated_action = (action_schema.name, holding_item, robot_room)
                        
                        # Build new plan
                        new_plan = plan + [instantiated_action]
                        
                        # Check if goal is reached
                        if goal.issubset(new_state_frozen):
                            return new_plan
                        
                        # Add to queue if not visited
                        if new_state_frozen not in visited_states:
                            visited_states.add(new_state_frozen)
                            queue.append((new_state_frozen, new_plan))
    
    # No plan found
    return None 