# centralizes all configuration management

# usage 
# this is imported into other files
# e.g crg = config.get_crg(crg_name)

# Think of the validators in config.py as checking the "skeleton" of your configuration. 
# Their job is to perform a quick "smoke test" at startup to answer one question: 
# "Is the configuration file so fundamentally broken that the application cannot possibly run?"