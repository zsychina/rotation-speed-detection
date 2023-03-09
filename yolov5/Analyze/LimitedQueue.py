class LimitedQueue:
    def __init__(self, n):
        self.queue = []
        self.max_size = n

    def push(self, item):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def __getitem__(self, i):
        return self.queue[i]

    def __len__(self):
        return len(self.queue)

if __name__ == '__main__':
    q = LimitedQueue(3)

    # add some items to the queue
    q.push(1)
    q.push(2)
    q.push(3)
    q.push(4)
    q.push(5)
    q.push(6)

    # access items by index
    print(q[0])
    print(q[1])

    # check the length of the queue
    print(len(q))
