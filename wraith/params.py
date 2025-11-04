class Params:
    """
    A class to hold and manage parameters for the galaxy processing pipeline.

    This class centralizes all configurable parameters, from user credentials and
    image specifications to IMFIT configurations and run-specific settings.
    An object of this class is instantiated and populated with values read from a
    configuration file, and then passed throughout the pipeline.

    Attributes:
        name (str): A general identifier for the run or sample, used for directory naming.
        user_id (str): The username for accessing data archives.
        user_pass (str): The password associated with the user_id.
        image_size (int): The size of the cutout image to be downloaded, in arcseconds.
        filter (str): The filter band to be used for the observations (e.g., 'i', 'g', 'r').
        use_def (int): A flag to determine whether to use the default IMFIT configuration.
                       Set to 1 to use the default, 0 to use a custom file.
        imfit_configfile (str): The path to a custom IMFIT configuration file.
                                This is used when `use_def` is 0.
        consecutive (int): A flag or counter for managing consecutive operations,
                         though its specific use may vary based on context.
        components (int): The number of components for the first IMFIT run.
        imfit_type (str): The type of IMFIT model to be used in the first run.
        components2 (int): The number of components for the second IMFIT run.
        imfit_type2 (str): The type of IMFIT model for the second run.
        components3 (int): The number of components for the third IMFIT run.
        imfit_type3 (str): The type of IMFIT model for the third run.
        components4 (int): The number of components for the fourth IMFIT run.
        imfit_type4 (str): The type of IMFIT model for the fourth run.
    """
    name = 'default'

    user_id = 'default'
    user_pass = 'default'

    image_size = 30
    filter = 'default'

    # imfit configs

    use_def = 1
    imfit_configfile = 'default'
    consecutive = 1

    # first run

    components = 1
    imfit_type = 'default'
    checks1 = 5

    # second_run

    components2 = 1
    imfit_type2 = 'default'
    checks2 = 5

    # third run

    components3 = 1
    imfit_type3 = 'default'
    checks3 = 5

    # fourth run

    components4 = 1
    imfit_type4 = 'default'
    checks4 = 5

    run_definitions = []