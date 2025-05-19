from robot_hmm import RobotHMM
from home_environment import HomeEnvironment
import numpy as np

def test_robot_hmm():
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
    
    # Define room observations
    room_observations = [
        'kitchen_sensed',
        'living_room_sensed',
        'bedroom_sensed',
        'bathroom_sensed',
        'unknown_sensed'
    ]
    
    # Test 1: Initialization
    print("Test 1: Initialize RobotHMM")
    hmm = RobotHMM(all_possible_locations, room_observations, env)
    
    print(f"Number of possible locations: {len(all_possible_locations)}")
    print(f"Initial belief state (first 5): {list(hmm.belief_state.items())[:5]}")
    
    # Check belief state sum to 1
    belief_sum = sum(hmm.belief_state.values())
    print(f"Sum of belief probabilities: {belief_sum}")
    
    assert abs(belief_sum - 1.0) < 1e-10, "Belief state should sum to 1"
    assert all(0 <= p <= 1 for p in hmm.belief_state.values()), "All probabilities should be between 0 and 1"
    initial_probability = 1.0 / len(all_possible_locations)
    assert all(abs(p - initial_probability) < 1e-10 for p in hmm.belief_state.values()), "Initial probabilities should be uniform"
    print("✓ Initialization test passed")
    print()
    
    # Test 2: Transition Model
    print("Test 2: Transition Model")
    prev_pos = (1, 1)  # Kitchen
    action = (1, 0)    # Move right
    
    # Expected next position
    expected_next_pos = (2, 1)  # Still kitchen
    prob_expected = hmm.get_transition_probability(prev_pos, action, expected_next_pos)
    print(f"Probability of moving as intended ({prev_pos} -> {expected_next_pos}): {prob_expected}")
    assert abs(prob_expected - hmm.transition_model_params['correct_move_prob']) < 1e-10, "Should match correct_move_prob"
    
    # Stay in place
    prob_stay = hmm.get_transition_probability(prev_pos, action, prev_pos)
    print(f"Probability of staying in place ({prev_pos} -> {prev_pos}): {prob_stay}")
    assert abs(prob_stay - hmm.transition_model_params['stay_prob']) < 1e-10, "Should match stay_prob"
    
    # Moving toward obstacle
    invalid_action = (-1, 0)  # Move left toward wall
    prob_stay_obstacle = hmm.get_transition_probability(prev_pos, invalid_action, prev_pos)
    print(f"Probability of staying when move would hit obstacle: {prob_stay_obstacle}")
    assert prob_stay_obstacle > hmm.transition_model_params['stay_prob'], "Should have increased stay probability when obstacle is in the way"
    print("✓ Transition Model test passed")
    print()
    
    # Test 3: Emission Model
    print("Test 3: Emission Model")
    kitchen_pos = (1, 1)
    
    # Correct observation
    prob_correct = hmm.get_emission_probability(kitchen_pos, 'kitchen_sensed')
    print(f"Probability of correct observation at kitchen: {prob_correct}")
    assert abs(prob_correct - hmm.emission_model_params['correct_room_sense_prob']) < 1e-10, "Should match correct_room_sense_prob"
    
    # Adjacent room observation
    prob_adjacent = hmm.get_emission_probability(kitchen_pos, 'living_room_sensed')
    print(f"Probability of adjacent room observation at kitchen: {prob_adjacent}")
    
    # Unknown observation
    prob_unknown = hmm.get_emission_probability(kitchen_pos, 'unknown_sensed')
    print(f"Probability of unknown observation at kitchen: {prob_unknown}")
    assert abs(prob_unknown - hmm.emission_model_params['unknown_sense_prob']) < 1e-10, "Should match unknown_sense_prob"
    print("✓ Emission Model test passed")
    print()
    
    # Test 4: Belief Update
    print("Test 4: Belief Update")
    # Focus on specific location to make testing easier
    focused_hmm = RobotHMM(all_possible_locations, room_observations, env)
    
    # Set initial belief state to be certain at (2, 1) (kitchen)
    certain_loc = (2, 1)
    for loc in all_possible_locations:
        focused_hmm.belief_state[loc] = 1.0 if loc == certain_loc else 0.0
    
    print(f"Initial certain belief at {certain_loc}")
    
    # Move right and get an observation from living room
    action = (1, 0)
    observation = 'living_room_sensed'
    
    # Update belief
    focused_hmm.update_belief(action, observation)
    
    # The belief should now be more concentrated on locations to the right
    print("Top 5 beliefs after update:")
    sorted_beliefs = sorted(focused_hmm.belief_state.items(), key=lambda x: x[1], reverse=True)
    for loc, prob in sorted_beliefs[:5]:
        print(f"Location {loc}: {prob:.6f}")
    
    # Check if belief state still sums to 1
    belief_sum = sum(focused_hmm.belief_state.values())
    print(f"Sum of updated belief probabilities: {belief_sum}")
    assert abs(belief_sum - 1.0) < 1e-10, "Updated belief state should sum to 1"
    
    # After moving right and getting living_room observation, we should have higher
    # probability for locations near the living room
    living_room_nearby = [(3, 1), (4, 1), (5, 1)]
    living_room_probs = [focused_hmm.belief_state.get(loc, 0) for loc in living_room_nearby]
    print(f"Probabilities near living room: {living_room_probs}")
    
    # At least one location near living room should have significant probability
    assert any(p > 0.1 for p in living_room_probs), "At least one location near living room should have significant probability"
    print("✓ Belief Update test passed")
    
    print("\nAll RobotHMM tests completed successfully!")

if __name__ == "__main__":
    test_robot_hmm() 