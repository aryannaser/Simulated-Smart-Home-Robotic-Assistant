from planner import forward_planner
from action_schema import ActionSchema

def test_planner():
    # Define sample action schemas
    
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
    
    # List of action schemas
    action_schemas = [goto_action, pickup_action, putdown_action]
    
    # Test 1: Initial state already satisfies goal
    print("Test 1: Initial state already satisfies goal")
    initial_state = {
        ('At', 'robot', 'kitchen'),
        ('At', 'cup', 'kitchen'),
        ('Holding', 'robot', 'nothing'),
        ('Connected', 'kitchen', 'living_room'),
        ('Connected', 'kitchen', 'bedroom'),
        ('Connected', 'living_room', 'kitchen'),
        ('Connected', 'bedroom', 'kitchen')
    }
    
    goal_state = {
        ('At', 'robot', 'kitchen')
    }
    
    plan = forward_planner(initial_state, goal_state, action_schemas)
    print(f"Plan: {plan}")
    
    assert plan == [], "Plan should be empty when goal is already satisfied"
    print("✓ Test 1 passed: Empty plan when goal already satisfied")
    print()
    
    # Test 2: Simple one-step plan (GoTo)
    print("Test 2: Simple one-step plan (GoTo)")
    initial_state = {
        ('At', 'robot', 'kitchen'),
        ('At', 'cup', 'kitchen'),
        ('Holding', 'robot', 'nothing'),
        ('Connected', 'kitchen', 'living_room'),
        ('Connected', 'kitchen', 'bedroom'),
        ('Connected', 'living_room', 'kitchen'),
        ('Connected', 'bedroom', 'kitchen')
    }
    
    goal_state = {
        ('At', 'robot', 'living_room')
    }
    
    plan = forward_planner(initial_state, goal_state, action_schemas)
    print(f"Plan: {plan}")
    
    assert plan is not None, "Plan should be found"
    assert len(plan) == 1, "Plan should have one step"
    assert plan[0][0] == "GoTo", "First action should be GoTo"
    assert plan[0][1] == "living_room", "GoTo parameter should be living_room"
    print("✓ Test 2 passed: One-step plan found")
    print()
    
    # Test 3: Simple two-step plan (GoTo and PickUp)
    print("Test 3: Simple two-step plan (GoTo and PickUp)")
    initial_state = {
        ('At', 'robot', 'kitchen'),
        ('At', 'cup', 'living_room'),
        ('Holding', 'robot', 'nothing'),
        ('Connected', 'kitchen', 'living_room'),
        ('Connected', 'kitchen', 'bedroom'),
        ('Connected', 'living_room', 'kitchen'),
        ('Connected', 'bedroom', 'kitchen')
    }
    
    goal_state = {
        ('Holding', 'robot', 'cup')
    }
    
    plan = forward_planner(initial_state, goal_state, action_schemas)
    print(f"Plan: {plan}")
    
    assert plan is not None, "Plan should be found"
    assert len(plan) == 2, "Plan should have two steps"
    assert plan[0][0] == "GoTo", "First action should be GoTo"
    assert plan[0][1] == "living_room", "GoTo parameter should be living_room"
    assert plan[1][0] == "PickUp", "Second action should be PickUp"
    assert plan[1][1] == "cup", "PickUp first parameter should be cup"
    print("✓ Test 3 passed: Two-step plan found")
    print()
    
    # Test 4: Unachievable goal
    print("Test 4: Unachievable goal")
    initial_state = {
        ('At', 'robot', 'kitchen'),
        ('At', 'cup', 'living_room'),
        ('Holding', 'robot', 'book'),  # Already holding something
        # Missing connection between kitchen and living room - making it unreachable
        ('Connected', 'kitchen', 'bedroom'),
        ('Connected', 'bedroom', 'kitchen')
    }
    
    goal_state = {
        ('Holding', 'robot', 'cup')  # Can't hold cup while holding book and can't reach living room
    }
    
    plan = forward_planner(initial_state, goal_state, action_schemas, max_depth=5)
    print(f"Plan: {plan}")
    
    assert plan is None, "No plan should be found for unachievable goal"
    print("✓ Test 4 passed: Unachievable goal recognized")
    
    print("\nAll planner tests completed successfully!")

if __name__ == "__main__":
    test_planner() 