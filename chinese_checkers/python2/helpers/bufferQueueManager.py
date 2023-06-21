from multiprocessing.managers import BaseManager

# custom manager to support custom classes, required to pass the replay buffer to all processes
class BufferQueueManager(BaseManager):
  # nothing
  pass