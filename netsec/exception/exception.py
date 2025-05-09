import sys

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message) # It's good practice to call super().__init__
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info() # Corrected: exc_info() is a function of sys module

        if exc_tb is not None: # Add check for exc_tb to prevent errors if no exception active
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = "Unknown"
            self.file_name = "Unknown"

    def __str__(self): # Corrected: Proper indentation for __str__
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )
