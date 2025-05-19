# Simulated Smart Home Robotic Assistant

A sophisticated Python simulation of a robotic assistant that navigates a home environment, locates and manipulates objects, and responds to natural language commands while handling uncertainty in sensing and movement.

## Project Overview

This project implements a complete simulation of a smart home robot that combines multiple AI and robotics techniques:

1. **Environment Modeling**: A grid-based representation of a home with rooms, obstacles, and items
2. **Probabilistic Localization**: HMM-based state estimation with noisy sensors and movement
3. **A* Pathfinding**: Efficient navigation through the home environment
4. **STRIPS-like Planning**: Automated planning for complex tasks using action schemas
5. **Natural Language Processing**: Basic command interpretation for user interactions

The robot can navigate between rooms, pick up and put down items, and execute multi-step plans like "fetch the cup from the kitchen and bring it to the living room" - all while dealing with uncertainty in its position and observations.

## Components

### Home Environment

The `HomeEnvironment` class represents the physical layout of the home using a 2D grid. It tracks:
- Room types (kitchen, living room, bedroom, bathroom)
- Obstacles (walls)
- Item locations
- Valid movement spaces

### Robot

The `Robot` class represents the robotic assistant with:
- Probabilistic belief state of its location
- Movement with realistic noise model (80% moves as intended, 10% stays in place, 10% drifts)
- Sensing with noise (70% correct room sensing, 15% adjacent room, 15% unknown)
- Item manipulation (pickup/putdown)
- Plan execution

### HMM-based Localization

The `RobotHMM` class implements a Hidden Markov Model for localization:
- Transition model accounting for action uncertainty
- Emission model for noisy sensor readings
- Belief state updates using Bayesian filtering

### A* Pathfinding

The `astar_search` function provides efficient pathfinding:
- Manhattan distance heuristic
- Obstacle avoidance
- Optimal path finding between any two points

### STRIPS-like Planning

The planning system uses:
- `ActionSchema` definitions for actions like GoTo, PickUp, PutDown
- `forward_planner` function for generating action sequences
- Preconditions and effects for state transitions
- Parameter binding for action instantiation

### Natural Language Interface

The system includes a basic natural language interface for commands like:
- "go to kitchen"
- "fetch cup"
- "fetch book to bedroom"

## Setup and Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/Simulated-Smart-Home-Robotic-Assistant.git
cd Simulated-Smart-Home-Robotic-Assistant

# No external dependencies beyond Python standard library and NumPy
pip install numpy
```

## Usage

### Interactive Mode

Run the main simulation to interact with the robot:

```bash
python main.py
```

You can issue commands like:
- `go to kitchen`
- `fetch cup`
- `fetch book to bedroom`
- `quit` to exit

### Automated Tests

Run the automated test suite to verify functionality:

```bash
python automated_test.py
```

This will run a series of predefined tests to verify the robot can:
1. Navigate to specific rooms
2. Pick up items
3. Execute multi-step plans (fetch and deliver objects)

## Technical Details

### Probabilistic Localization

The robot maintains a belief state over all possible locations. When it takes actions or receives sensor readings, it updates this belief state using:

1. **Transition update**: P(x_t | x_{t-1}, a_t) - Models how actions affect the robot's state
2. **Sensor update**: P(z_t | x_t) - Models how sensor readings relate to the actual state

This allows the robot to maintain an estimate of its position even with noisy sensors and actions.

### Planning System

The planning system uses a forward search approach:
1. Start with the current state
2. Generate possible actions and their resulting states
3. Use breadth-first search to find a path to the goal state
4. Return the sequence of actions leading to the goal

Actions have:
- **Preconditions**: What must be true to apply the action
- **Effects**: How the action changes the world state

### Room Navigation

The robot navigates between rooms using:
1. A* pathfinding to find the optimal path
2. Special handling for doorways and transitions between rooms
3. Recovery mechanisms when navigation fails due to noise

### Error Handling

The system includes robust error handling:
- Recovery from navigation failures
- Replanning when the robot gets lost
- Special handling for difficult room transitions

## Project Structure

- `home_environment.py`: Environment representation
- `robot.py`: Robot implementation with movement and item manipulation
- `robot_hmm.py`: HMM implementation for probabilistic localization
- `astar_search.py`: A* pathfinding algorithm
- `action_schema.py`: STRIPS-like action schema definitions
- `planner.py`: Forward planning algorithm
- `main.py`: Main simulation loop with user interaction
- `automated_test.py`: Automated test suite

## Testing

The project includes comprehensive tests:
- Unit tests for individual components
- Integration tests for combined functionality
- End-to-end tests for complete task execution

Run specific test files:
```bash
python test_robot.py
python test_home_environment.py
python test_astar_search.py
python test_robot_hmm.py
python test_robot_with_hmm.py
python test_action_schema.py
python test_planner.py
```

## Future Improvements

Potential enhancements for future versions:
- More sophisticated natural language understanding
- Better handling of partially observable environments
- Learning environment dynamics from experience
- Multi-agent coordination
- Visual recognition of objects and rooms

## License

This project is licensed under the terms of the included LICENSE file. 