class ReplayBufer:

    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)

    def append(self, data):
        """Append data to back of deque"""
        if len(data) < self.max_size:
            self.memory.popleft()
        
        self.memory.append(data)
