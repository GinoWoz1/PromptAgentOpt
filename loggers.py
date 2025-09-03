# The RichHandler is the key component for making the logs look good.
# It comes from the rich library, a popular tool for creating beautiful formatting in the terminal.
# Think of RichHandler as a translator. 
# It intercepts the standard, plain-text log messages from Python's logging system and reformats them with helpful features before printing them to the console. These features include:
# Color-coding by log level (e.g., INFO is plain, WARNING is yellow, ERROR is red).
# Syntax highlighting for data structures.
# Clean, readable formatting for Python tracebacks (error messages), which is enabled by rich_tracebacks=True.