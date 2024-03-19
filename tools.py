import sys

class Page:
    def __init__(self, _id):
        self.id = _id

    def __eq__ (self, other):
        return self.id == other.id
    
    def __str__(self) -> str:
        return str(self.id)


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    def is_empty(self):
        return len(self.stack) == 0

    def peek(self):
        return self.stack[-1]

    def size(self):
        return len(self.stack)


class Queue:
    def __init__(self):
        self.body = [None] * 10
        self.front = 0    # index of first element, but 0 if empty
        self.size = 0
    
    def __str__(self) -> str:
        output = ""
        i = self.front
        if self.front < ((self.front+self.size) % len(self.body)):
            while i < ((self.front+self.size) % len(self.body)):
                output += str(self.body[i]) + " "
                i = i+1
        else:
            while i < len(self.body):
                output += str(self.body[i]) + " "
                i = i+1
            i = 0
            while i < ((self.front+self.size) % len(self.body)):
                output += str(self.body[i]) + " "
                i = i+1
        return output

    def get_size(self):
        return sys.getsizeof(self.body)

    def shrink(self):
        oldbody = self.body
        self.body = [None] * (self.size + 5)
        for i in range(self.size + 5):
            self.body[i] = oldbody[(self.front + i)% len(oldbody)]
        self.front = 0

    def grow(self):
        oldbody = self.body
        self.body = [None] * (2*self.size)
        for i in range(self.size):
            self.body[i] = oldbody[(self.front + i)% len(oldbody)]
        self.front = 0

    def enqueue(self,item):
        if self.size == 0:
            self.body[0] = item      # assumes an empty queue has head at 0
            self.front = 0
            self.size = 1
        else:
            self.body[(self.front+self.size) % len(self.body)] = item
            self.size = self.size + 1
            if self.size == len(self.body):  # list is now full
                self.grow()                  # so grow it ready for next enqueue


    def dequeue(self):
        if self.size == 0:     # empty queue
            return None
        item = self.body[self.front]
        self.body[self.front] = None
        if self.size == 1:
            self.front = 0
            self.size = 0
        elif self.front == len(self.body) - 1:
            self.front = 0
            self.size = self.size - 1
        else:
            self.front = (self.front + 1) % len(self.body)
            self.size = self.size - 1
        if self.size < (len(self.body) // 4) and len(self.body) > 10:
            self.shrink()
        return item

    def length(self) -> int:
        return self.size

    def first(self):
        return self.body[self.front]

    def is_empty(self) -> bool:
        return self.size == 0
    
    def iterator(self) -> iter:
        """Returns the items of the queue in order as an iterator."""
        i = self.front
        while i != (self.front+self.size) % len(self.body):
            if self.body[i] != None:
                yield self.body[i]
            i = (i+1) % len(self.body)
