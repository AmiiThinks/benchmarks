import random
import collections


# Replay buffer class with deque
class ReplayBuffer:
  def __init__(self, capacity, replace_old_data_probability, train_percentage, sample_size):
    self.capacity = capacity # capacity of the buffer
    self.probability = replace_old_data_probability # probability of replacing old data
    self.buffer = collections.deque(maxlen=capacity) # buffer
    self.train_percentage = train_percentage # percentage of new data to be added to the buffer before training
    self.new_item_counter = 0 # counts the number of new items added to the buffer
    self.sample_size = sample_size # sample size for training

  def push(self, experience):
    if len(self.buffer) > self.capacity:  # replace old data with probablity
      if random.uniform(0, 1) < (1-self.probability):
        self.buffer.append(experience)
    else:
      self.buffer.append(experience) # add new data
    self.new_item_counter += 1

  def is_ready_for_training(self):
    new_item_percentage = self.new_item_counter / (self.capacity-1)
    if new_item_percentage > self.train_percentage : 
      return True
    return False

  def reset_new_item_counter(self):
    self.new_item_counter = 0

  def sample(self):
    if len(self.buffer) < self.sample_size: # return all data if sample size is greater than buffer size
      return random.sample(self.buffer, len(self.buffer))
    return random.sample(self.buffer, self.sample_size) # return sample of data


# Replay buffer class with multiprocessing queue
# class ReplayBufferWithQueue:
#   def __init__(self, queue, capacity, new_data_percentage):
#     self.capacity = capacity
#     self.new_data_percentage = new_data_percentage
#     self.memory = queue
#     self.new_item_counter = 0
#     # self.lock = Lock()

#   def push(self, data):
#     if self.memory.full():
#       self.memory.get()  # remove the last element

#     self.memory.put(data)
#     self.new_item_counter += 1

#   def is_ready_for_training(self):
#     if self.memory.full():
#       if self.new_item_counter / self.capacity > self.new_data_percentage:
#         return True
#       return False
#     return False

#   def reset_new_item_counter(self):
#     self.new_item_counter = 0

#   def sample(self):
#     # self.lock.acquire() # acquire the lock
    
#     experiences = []
#     try:
#       while True:
#         experiences.append(self.memory.get(timeout=0.1))
#     except:
#       pass

      
#     # put all elements back in the queue
#     for experience in experiences:
#       self.memory.put(experience)

#     # self.lock.release() # release the lock
#     return random.sample(experiences, len(experiences))




if __name__ == "__main__":
    memory = ReplayBuffer(capacity=1000, replace_old_data_probability=0.05, train_percentage=0.2, sample_size=50)
    for i in range(1000):
        if memory.is_ready_for_training():
            print(len(memory.sample()))
            memory.reset_new_item_counter()
        memory.push(i)