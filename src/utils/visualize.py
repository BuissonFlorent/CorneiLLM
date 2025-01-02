from typing import List, Union

def display_tokens(tokens: List[str], max_tokens: Union[int, None] = None) -> str:
    """Display tokens in a readable format with indentation.
    
    Args:
        tokens (List[str]): List of tokens to display
        max_tokens (int, optional): Maximum number of tokens to display
        
    Returns:
        str: Formatted string with proper indentation
    """
    # Handle max_tokens
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    
    indent = 0
    result = []
    
    for token in tokens:
        # Decrease indent for closing tags
        if token.startswith('</'):
            indent -= 1
            
        # Add token with proper indentation
        result.append('    ' * indent + token)
        
        # Increase indent for opening tags
        if token.startswith('<') and not token.startswith('</'):
            if not token.startswith('</') and not token in ['<PAD>', '<UNK>']:
                indent += 1
                
    return '\n'.join(result) 