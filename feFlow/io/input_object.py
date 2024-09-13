'''
Class for reading input file
'''
import sys

class InputObject:

    """
    :param input_file: Path to the input file
    :type input_file: str
    """


    def __init__(self, input_file):


        self.input_file = input_file
        fid = open(input_file, "r")
        self.input_dict = {}
        for line in fid:
            line = line.strip()
            if line.startswith('#') or line=='':
                continue
            key = line.split("=")[0].strip()
            value = line.split("=")[1].strip()
            if key == "List":
                # Handle list type input
                key = value
                value = []
                while line.split("=")[0].strip() != "EndList":
                    # Each entry in the input list is an entry in values
                    line = fid.readline().strip()
                    if line.startswith('#') or line=='':
                        continue
                    list_dict = {}
                    if line.split("=")[0].strip() != "EndList":
                        lineList = line.split(";")
                        for entry in lineList:
                            entry_key = entry.split("=")[0].strip()
                            entry_value = self._parse_input(entry.split("=")[1].strip())
                            list_dict[entry_key] = entry_value
                        value.append(list_dict)
                line = fid.readline() # read out the EndList line
            else:
                value = self._parse_input(value)
            self.input_dict[key] = value
        fid.close()

    def __call__(self, key, required=False, warn=True):

        output = None
        try:
            output = self.input_dict[key]
        except KeyError:
            if required:
                raise Exception("feFlow ERROR: MISSING REQUIRED INPUT: \'%s\'" % key)
            else:
                if warn:
                    print("feFlow WARNING: KEY/VALUE PAIR NOT FOUND: \'%s\'" % key )
        return output

    def _parse_input(self, input_str):

        """
        Parse an input string based on specific rules:

        1. If the input string is "true" or "false" (case-insensitive), return the corresponding boolean value.
        2. If the input string can be converted to a number, return it as an integer (if it's an integer) or a float (if it's a float).
        3. If the input string is a tuple in the format "(element1, element2, ...)", parse the elements inside the tuple:
           - If all elements can be converted to numbers, return a tuple of integers or floats as appropriate.
           - If any element cannot be converted to a number, return a tuple with elements as strings.
        4. If none of the above conditions are met, return the input string as is.

        Parameters:
        input_str (str): The input string to be parsed.

        Returns:
        bool, int, float, str, or tuple: The parsed value or the original input string.
        """

        # Remove leading and trailing white spaces
        input_str = input_str.strip()

        # Check if the input string is "true" or "false" and return the corresponding boolean
        if input_str.lower() == "true":
            return True
        elif input_str.lower() == "false":
            return False

        try:
            # Try to convert the input string to a float
            num = float(input_str)
            # If it's an integer, return as int, otherwise return as float
            return int(num) if num.is_integer() else num
        except ValueError:
            # If it's not a number, check if it's a tuple
            if input_str.startswith('(') and input_str.endswith(')'):
                parsed_elements = self._parse_tuple_elements(input_str)
                any_elements_as_strings = any(isinstance(elem, str) for elem in parsed_elements)
                return tuple(parsed_elements)

                # if any_elements_as_strings:
                #     return tuple(str(elem) for elem in parsed_elements)
                # else:
                #     return tuple(parsed_elements)
            else:
                # If it's not a number, boolean, or a tuple, return the original string
                return input_str

    def _parse_tuple_elements(self, tuple_str):

        """
        Parse the elements of a string representing a tuple.

        This function takes a string containing elements in tuple format and attempts to parse each element.

        Parameters:
        tuple_str (str): The string representing a tuple (e.g., "(element1, element2, ...)") to be parsed.

        Returns:
        list: A list containing the parsed elements. Elements are either integers, floats, bool, or strings.
        """

        try:
            # Try to parse the contents of the tuple
            tuple_str = tuple_str[1:-1]
            elements = tuple(element.strip() for element in tuple_str.split(','))
            parsed_elements = []

            for elem in elements:
                parsed_elements.append(self._parse_input(elem))
                # try:
                #     parsed_elements.append(int(elem))
                # except ValueError:
                #     try:
                #         parsed_elements.append(float(elem))
                #     except ValueError:
                #         parsed_elements.append(elem)
            return parsed_elements

        except ValueError:
            # Handle a specific ValueError (e.g., when conversion fails)
            return [tuple_str]

    def dump(self):

        """
        Print out all of the entries
        """

        EOL = '# -------------------------------------'
        for key, value in self.input_dict.items():
            print('\n'+EOL)
            if type(value) is list:
                strOut = ('%s = \n' %(key))
                i = 0
                for v in value:
                    if i == len(value)-1:
                        str_vals = ('%d %s' %(i,v))
                    else:
                        str_vals = ('%d %s\n' %(i, v))
                    strOut = strOut + str_vals
                    i+=1
                print(strOut)
            else:
                print('%s = %s' %(key, value))
            print(EOL)











