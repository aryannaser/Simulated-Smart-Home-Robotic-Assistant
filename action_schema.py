class ActionSchema:
    def __init__(self, name, parameters, preconditions, add_effects, delete_effects):
        """
        Initialize an action schema for planning.
        
        Args:
            name: Name of the action (string)
            parameters: Tuple of parameter names (strings)
            preconditions: Set of predicate tuples that must be true for the action to be applicable
            add_effects: Set of predicate tuples that become true after the action
            delete_effects: Set of predicate tuples that become false after the action
        
        Note: A predicate tuple could be like ('At', 'robot', 'kitchen') or ('Holding', 'robot', 'cup')
        """
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.add_effects = add_effects
        self.delete_effects = delete_effects
    
    def __str__(self):
        """String representation of the action schema."""
        param_str = ", ".join(self.parameters)
        return f"{self.name}({param_str})"
    
    def __repr__(self):
        """Detailed representation of the action schema."""
        return (f"ActionSchema(name='{self.name}', parameters={self.parameters}, "
                f"preconditions={self.preconditions}, add_effects={self.add_effects}, "
                f"delete_effects={self.delete_effects})") 