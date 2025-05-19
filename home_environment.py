import numpy as np

class HomeEnvironment:
    def __init__(self, grid_layout, item_locations):
        """
        Initialize the home environment with a grid layout and item locations.
        
        Args:
            grid_layout: 2D list or numpy array representing the environment
                        (0: empty, 1: obstacle, string: room type)
            item_locations: Dictionary mapping item names to (x, y) coordinates
        """
        self.grid = np.array(grid_layout, dtype=object)
        self.item_locations = item_locations.copy()
        self.height, self.width = self.grid.shape
    
    def is_obstacle(self, x, y):
        """
        Check if the given coordinates represent an obstacle or are out of bounds.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            True if (x, y) is an obstacle or out of bounds, False otherwise
        """
        # Check if out of bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        
        # Check if obstacle
        return self.grid[y, x] == 1
    
    def get_room_type(self, x, y):
        """
        Get the room type at the given coordinates.
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Room type string if cell contains a room type, None otherwise
        """
        if self.is_obstacle(x, y):
            return None
        
        cell_value = self.grid[y, x]
        if isinstance(cell_value, str):
            return cell_value
        return None
    
    def get_item_location(self, item_name):
        """
        Get the location of the specified item.
        
        Args:
            item_name: Name of the item to find
            
        Returns:
            (x, y) tuple if item exists, None otherwise
        """
        return self.item_locations.get(item_name)
    
    def update_item_location(self, item_name, new_location):
        """
        Update the location of an item.
        
        Args:
            item_name: Name of the item to update
            new_location: New (x, y) coordinates or None if item is held
        """
        self.item_locations[item_name] = new_location
    
    def get_valid_neighbors(self, x, y):
        """
        Get valid non-obstacle neighboring cells.
        
        Args:
            x, y: Coordinates to find neighbors for
            
        Returns:
            List of (nx, ny) tuples representing valid neighbors
        """
        potential_neighbors = [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1)
        ]
        
        return [(nx, ny) for nx, ny in potential_neighbors if not self.is_obstacle(nx, ny)] 