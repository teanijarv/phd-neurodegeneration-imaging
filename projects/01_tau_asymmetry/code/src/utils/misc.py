import os

def find_src_directory(start_path=None, max_levels=5):
    if start_path is None:
        start_path = os.getcwd()
    
    current_path = os.path.abspath(start_path)
    
    # Search up to max_levels directories up
    for _ in range(max_levels):
        # Check if 'src' exists in current directory
        potential_src = os.path.join(current_path, 'src')
        if os.path.isdir(potential_src):
            return potential_src
        
        # Move one level up
        parent = os.path.dirname(current_path)
        if parent == current_path:  # Reached root directory
            break
        current_path = parent
    
    raise FileNotFoundError("src directory not found within specified levels")