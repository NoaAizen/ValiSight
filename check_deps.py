import importlib.util
print('pyserial', bool(importlib.util.find_spec('serial')))
print('cv2', bool(importlib.util.find_spec('cv2')))
print('numpy', bool(importlib.util.find_spec('numpy')))
