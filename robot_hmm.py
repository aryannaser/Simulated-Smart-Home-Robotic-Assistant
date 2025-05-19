import numpy as np

class RobotHMM:
    def __init__(self, all_possible_locations, room_observations, environment):
        """
        Initialize RobotHMM for probabilistic localization.
        
        Args:
            all_possible_locations: List of (x, y) tuples representing all valid non-obstacle coordinates.
            room_observations: List of possible sensor readings, e.g., ['kitchen_sensed', 'living_room_sensed', 'unknown_sensed'].
            environment: Instance of HomeEnvironment.
        """
        self.all_possible_locations = all_possible_locations
        self.room_observations = room_observations
        self.environment = environment
        
        # Initialize belief state with uniform probability
        self.belief_state = {}
        initial_prob = 1.0 / len(all_possible_locations)
        for loc in all_possible_locations:
            self.belief_state[loc] = initial_prob
        
        # Set transition model parameters
        self.transition_model_params = {
            'correct_move_prob': 0.8,  # Probability of moving as intended
            'stay_prob': 0.1,          # Probability of staying in the same place
            'slip_prob': 0.1           # Probability of slipping to an unintended neighbor
        }
        
        # Set emission model parameters
        self.emission_model_params = {
            'correct_room_sense_prob': 0.7,     # Probability of correctly sensing the room type
            'wrong_adj_room_sense_prob': 0.15,  # Probability of sensing an adjacent room type
            'unknown_sense_prob': 0.15          # Probability of getting an unknown/error reading
        }
    
    def get_transition_probability(self, prev_pos, intended_action_vector, next_pos):
        """
        Calculate transition probability P(next_pos | prev_pos, intended_action).
        
        Args:
            prev_pos: Previous position (x, y).
            intended_action_vector: Intended movement vector (dx, dy).
            next_pos: Next position (x, y).
            
        Returns:
            Probability of transitioning from prev_pos to next_pos given intended_action_vector.
        """
        # Calculate expected next position based on intended action
        expected_next_x = prev_pos[0] + intended_action_vector[0]
        expected_next_y = prev_pos[1] + intended_action_vector[1]
        expected_next_pos = (expected_next_x, expected_next_y)
        
        # Check if expected next position is valid (not an obstacle or out of bounds)
        expected_pos_valid = not self.environment.is_obstacle(expected_next_x, expected_next_y)
        
        # Get valid neighbors of previous position
        valid_neighbors = self.environment.get_valid_neighbors(prev_pos[0], prev_pos[1])
        
        # If expected position is not valid, adjust probabilities
        if not expected_pos_valid:
            # If attempting to move to an obstacle, higher chance of staying in place
            if next_pos == prev_pos:
                return self.transition_model_params['correct_move_prob'] + self.transition_model_params['stay_prob']
            # Distribute slip_prob among valid neighbors
            elif next_pos in valid_neighbors:
                return self.transition_model_params['slip_prob'] / len(valid_neighbors)
            else:
                return 0.0
        
        # Handle the case where the expected position is valid
        if next_pos == expected_next_pos:
            return self.transition_model_params['correct_move_prob']
        elif next_pos == prev_pos:
            return self.transition_model_params['stay_prob']
        elif next_pos in valid_neighbors:
            # Distribute slip_prob among valid neighbors except the expected next pos
            valid_unintended_neighbors = [n for n in valid_neighbors if n != expected_next_pos]
            if valid_unintended_neighbors:
                return self.transition_model_params['slip_prob'] / len(valid_unintended_neighbors)
        
        return 0.0
    
    def get_emission_probability(self, true_pos, observation):
        """
        Calculate emission probability P(observation | true_pos).
        
        Args:
            true_pos: True position (x, y).
            observation: Observation string (e.g., 'kitchen_sensed').
            
        Returns:
            Probability of receiving the observation when at true_pos.
        """
        # Get the room type at the true position
        true_room_type = self.environment.get_room_type(true_pos[0], true_pos[1])
        
        # If not in a room (empty space or obstacle), use simpler model
        if true_room_type is None:
            # Higher probability of getting 'unknown_sensed'
            if observation == 'unknown_sensed':
                return 0.8
            else:
                # Distribute remaining probability among other observations
                return 0.2 / (len(self.room_observations) - 1) if len(self.room_observations) > 1 else 0.2
        
        # Expected observation for this room
        expected_observation = f"{true_room_type}_sensed"
        
        # Get adjacent room types
        adjacent_positions = self.environment.get_valid_neighbors(true_pos[0], true_pos[1])
        adjacent_room_types = set()
        for adj_pos in adjacent_positions:
            room_type = self.environment.get_room_type(adj_pos[0], adj_pos[1])
            if room_type is not None:
                adjacent_room_types.add(room_type)
        
        # Create set of expected observations for adjacent rooms
        adjacent_observations = {f"{room_type}_sensed" for room_type in adjacent_room_types}
        
        # Calculate emission probability
        if observation == expected_observation:
            return self.emission_model_params['correct_room_sense_prob']
        elif observation in adjacent_observations:
            return self.emission_model_params['wrong_adj_room_sense_prob'] / len(adjacent_observations) if adjacent_observations else 0.0
        elif observation == 'unknown_sensed':
            return self.emission_model_params['unknown_sense_prob']
        else:
            # Distribute remaining probability among other non-adjacent observations
            other_observations = [obs for obs in self.room_observations 
                               if obs != expected_observation 
                               and obs != 'unknown_sensed'
                               and obs not in adjacent_observations]
            return (1.0 - self.emission_model_params['correct_room_sense_prob'] 
                   - self.emission_model_params['unknown_sense_prob']
                   - (self.emission_model_params['wrong_adj_room_sense_prob'] if adjacent_observations else 0.0)) / len(other_observations) if other_observations else 0.0
    
    def update_belief(self, intended_action_vector, observation_received):
        """
        Update the belief state based on action and observation (forward algorithm).
        
        Args:
            intended_action_vector: Intended action vector (dx, dy).
            observation_received: Observation string received from the environment.
        """
        # Step 1: Prediction step (apply transition model)
        predicted_belief = {}
        
        for current_loc in self.all_possible_locations:
            # Sum over all possible previous locations
            predicted_belief[current_loc] = 0.0
            for prev_loc in self.all_possible_locations:
                transition_prob = self.get_transition_probability(prev_loc, intended_action_vector, current_loc)
                predicted_belief[current_loc] += transition_prob * self.belief_state[prev_loc]
        
        # Step 2: Update step (apply emission model)
        new_belief = {}
        total_probability = 0.0
        
        for current_loc in self.all_possible_locations:
            emission_prob = self.get_emission_probability(current_loc, observation_received)
            new_belief[current_loc] = emission_prob * predicted_belief[current_loc]
            total_probability += new_belief[current_loc]
        
        # Step 3: Normalize
        if total_probability > 0:
            for loc in self.all_possible_locations:
                new_belief[loc] /= total_probability
        
        # Update belief state
        self.belief_state = new_belief 