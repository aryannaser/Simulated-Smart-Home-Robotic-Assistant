from action_schema import ActionSchema

def test_action_schema():
    # Define sample action schemas for testing
    
    # GoTo action schema
    goto_name = "GoTo"
    goto_parameters = ('room',)
    goto_preconditions = {
        ('At', 'robot', 'current_room'),
        ('Connected', 'current_room', 'room')
    }
    goto_add_effects = {
        ('At', 'robot', 'room')
    }
    goto_delete_effects = {
        ('At', 'robot', 'current_room')
    }
    
    goto_action = ActionSchema(
        goto_name,
        goto_parameters,
        goto_preconditions,
        goto_add_effects,
        goto_delete_effects
    )
    
    # PickUp action schema
    pickup_name = "PickUp"
    pickup_parameters = ('item', 'room')
    pickup_preconditions = {
        ('At', 'robot', 'room'),
        ('At', 'item', 'room'),
        ('Holding', 'robot', 'nothing')
    }
    pickup_add_effects = {
        ('Holding', 'robot', 'item')
    }
    pickup_delete_effects = {
        ('At', 'item', 'room'),
        ('Holding', 'robot', 'nothing')
    }
    
    pickup_action = ActionSchema(
        pickup_name,
        pickup_parameters,
        pickup_preconditions,
        pickup_add_effects,
        pickup_delete_effects
    )
    
    # PutDown action schema
    putdown_name = "PutDown"
    putdown_parameters = ('item', 'room')
    putdown_preconditions = {
        ('At', 'robot', 'room'),
        ('Holding', 'robot', 'item')
    }
    putdown_add_effects = {
        ('At', 'item', 'room'),
        ('Holding', 'robot', 'nothing')
    }
    putdown_delete_effects = {
        ('Holding', 'robot', 'item')
    }
    
    putdown_action = ActionSchema(
        putdown_name,
        putdown_parameters,
        putdown_preconditions,
        putdown_add_effects,
        putdown_delete_effects
    )
    
    # Print action schema details
    print("GoTo action schema:")
    print(f"Name: {goto_action.name}")
    print(f"Parameters: {goto_action.parameters}")
    print(f"Preconditions: {goto_action.preconditions}")
    print(f"Add Effects: {goto_action.add_effects}")
    print(f"Delete Effects: {goto_action.delete_effects}")
    print(f"String representation: {goto_action}")
    print()
    
    print("PickUp action schema:")
    print(f"Name: {pickup_action.name}")
    print(f"Parameters: {pickup_action.parameters}")
    print(f"Preconditions: {pickup_action.preconditions}")
    print(f"Add Effects: {pickup_action.add_effects}")
    print(f"Delete Effects: {pickup_action.delete_effects}")
    print(f"String representation: {pickup_action}")
    print()
    
    print("PutDown action schema:")
    print(f"Name: {putdown_action.name}")
    print(f"Parameters: {putdown_action.parameters}")
    print(f"Preconditions: {putdown_action.preconditions}")
    print(f"Add Effects: {putdown_action.add_effects}")
    print(f"Delete Effects: {putdown_action.delete_effects}")
    print(f"String representation: {putdown_action}")
    
    print("\nAll action schema tests completed successfully!")

if __name__ == "__main__":
    test_action_schema() 