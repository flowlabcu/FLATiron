InputObject
----------------

This class is a general purpose input file handler which parse input files. Each entry in the input file will consist two types of lines - these are the value assignment and comments. 

#. Comment lines always start with ``#`` and are ignored.

#. Value assignment line is formatted as ``name = value``.Anything to the left of the ``=`` is considered the variable name, and will be read in as string. Anything to the right of the ``=`` is considered the variable value. This class will automatically parse the value into simple python type. Available types are ``int``, ``float``, ``str``, ``bool``, or a ``tuple`` of the aforementioned type.

#. **Note that this class internally stores the name/value pair in a python dictionary, therefore any duplicate names in the input file will be overwritten in the internal dicitonary**

============================================
Example input file
============================================

.. code-block::

    # Any line starting with the # is considered comment, and is ignored

    # Variable name is anything to the left of the `=`, so we can have space in the name.
    package name = feFlow

    # These values are automatically parsed into the appopriate data type
    i = 1
    f = 1.5
    e = 1e2

    # Boolean variables are signified as true or false and will be parsed accordingly
    bt = true
    bf = false

    # We can also assign tuple for a collection of values. Each member of the tuple will be
    # tuple are signified as (a, b, ...)
    # automatically parsed to the appopriate type
    t = (1, 1e2, hello, true)


============================================
Example class initialization
============================================

.. code-block:: python

    from feFlow.io import InputObject
    input_object = InputObject('example_input_file.inp') # where the ``example_input_file.inp`` is the file above

    package_name = input_object("package name") # return string "feFlow"
    i = input_object("i") # return int 1
    f = input_object("f") # return float 1.5
    e = input_object("e") # return float 100.0
    bt = input_object("bt") # return boolean True
    bf = input_object("bf") # return boolean False
    t = input_object("t") # return tuple (1, 100.0, "hello", True)





===================
Class definition
===================

.. autoclass:: feFlow.io.InputObject
    :members:
    :private-members:



