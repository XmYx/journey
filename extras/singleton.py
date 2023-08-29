

# class Singleton:


#     _instance = None
#
#     @staticmethod
#     def getInstance():
#         """Static access method."""
#         if Singleton._instance is None:
#             Singleton()
#         return Singleton._instance
#
#     def __init__(self):
#         """Virtually private constructor."""
#         if Singleton._instance is not None:
#             raise Exception("This class is a singleton!")
#         else:
#             Singleton._instance = self
#             # Add your loaded modules or data here.
#             self.data = {}
#             self.base_loaded = None
