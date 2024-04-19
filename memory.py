# visualistion
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"   # hide the pygame support prompt, which is shouldn't be in the library in the first place...

from visualise import Visualiser
from math import log2
from errors import SecurityError, SimulatedMemoryError, InvalidMemoryRequestError, invalidInitialisationParameterError, MissingBlockFrameError, AllocatedBlockFrameError
from tools import Queue

# Future things to do:
# Change the way memory is written to (use address only, ideally)
# Use BlockFrames in the place of blocks and create a co-dependancy between the two for methods
# Add more stringent checks for security
# Use better memory management algorithms
# Intergate with the process manager + more?

# All sizes are in KB

MAIN_MEMORY_SIZE = 1 << 12  # 4096 KB
PAGE_SIZE = 1 << 2  # 4 KB
MIN_BLOCK_SIZE = PAGE_SIZE << 2 # 16 KB
NUM_BLOCK_SIZES = int(log2(MAIN_MEMORY_SIZE // MIN_BLOCK_SIZE)) + 1 # must include biggest and smallest size, so +1

if MAIN_MEMORY_SIZE < PAGE_SIZE: raise invalidInitialisationParameterError("Main memory must be larger than page size")
if MAIN_MEMORY_SIZE % PAGE_SIZE != 0: raise invalidInitialisationParameterError("Main memory must be divisible by page size")
if MIN_BLOCK_SIZE < PAGE_SIZE: raise invalidInitialisationParameterError("Minimum block size must be larger than page size")


vis = Visualiser(MAIN_MEMORY_SIZE // MIN_BLOCK_SIZE)

class ProcessContainer:
    def __init__(self):
        self._allocated_block = None
        self._start_range = None
        self._end_range = None
    
    def set_block(self):
        ...

class BlockFrame:
    def __init__(self, address: int, size: int):
        assert address % size == 0, "Block out of alignment"
        self.block = None
        self._size = size
        self.address = address
        self.free = True
    
    def __str__(self) -> str:
        return f"ID: {id(self)}, Address: {self.address}, Size: {self._size}, Free: {self.free}, Block: [{self.block}]"
    
    def set_block(self, block: 'Block'):
        """Sets the block for the frame and links both directions."""
        self.block = block
        block.set_frame(self)
        self.free = False
    
    def remove_block(self):
        """Removes the block from the frame and unlinks both directions."""
        self.block.remove_frame()
        self.block = None
        self.free = True
    
    def get_size(self) -> int:
        """Gets the size of the block"""
        return self._size
    
    def get_buddy(self) -> int:
        """Gets the address of the buddy block"""
        return self.address ^ self._size  # address XOR size = buddy
    
    def get_parent(self) -> int:
        """Gets the address of the parent block"""
        return self.address & ~((self._size << 1) - 1) # address AND NOT (2*size-1) = parent
    
    def get_children(self) -> list[int]:
        """Gets the address of the two child blocks upon splitting the block."""
        return [self.address, self.address + (self._size >> 1)] # address, address + size = children


class Block:
    def __init__(self, parent_block_frame: BlockFrame):
        self.set_frame(parent_block_frame)
        self.IsInMemory = False

    def __str__(self) -> str:
        return f"ID: {id(self)}, Address: {self.address}, Size: {self._size}, IsInMemory: {self.IsInMemory}"
    
    def set_frame(self, frame: BlockFrame):
        """Sets the frame for the block."""
        self.block_frame = frame
        self.IsInMemory = True
        self.address = frame.address
        self._size = frame.get_size()
    
    def remove_frame(self):
        """Removes the frame from the block."""
        self.block_frame = None
        self.IsInMemory = False
        self.address = None
        # we need to keep the size even when it's not in main memory

    def get_size(self) -> int:
        """Gets the size of the block"""
        return self._size
    
    def get_buddy(self) -> int:
        """Gets the address of the buddy block"""
        return self.block_frame.get_buddy()
    
    def get_parent(self) -> int:
        """Gets the address of the parent block"""
        return self.block_frame.get_parent()
    
    def get_children(self) -> list[int]:
        """Gets the address of the two child blocks upon splitting the block."""
        return self.block_frame.get_children()

class Memory:
    def __init__(self):
        self.memory = [None] * MAIN_MEMORY_SIZE
        self.disk_memory = [None] * MAIN_MEMORY_SIZE    # should be much larger, but no point simulating too much for proof-of-concept
        self.disk_lookup = {}                           # Should be handled as virtual memory, but it is too long to implement in a lab, so we just store the addresses for the moved blocks.
        self.next_disk_mem_free = 0                     # Next free address in the disk. Very inefficient, probably should be looping, but it is a quick solution to evicted blocks. Ideally a similar memory manager, but simpler should be used.
        self.FIFOBlocks = Queue()                       # FIFO but with entire blocks instead of pages
        self.free_block_frames = [[] for _ in range(NUM_BLOCK_SIZES)]
        self.allocated_block_frames : dict[int, BlockFrame] = {}

        init_block = self.add_block_frame(0, MAIN_MEMORY_SIZE)
        self.free_block_frames[self.size_to_index(init_block.get_size())].append(init_block)


        # For stats
        self.page_faults = 0
        self.page_ops = 0

        self.print_debug = False        # Quick enabling of debug

    def debug(self, item):
        """Prints the debug message if the debug flag is set."""
        if self.print_debug and type(item) == Memory:
            item.__str__()  # visualise
    
    def stats(self) -> tuple[float, float]:
        """Returns the stats of the memory. Returns the page fault rate and the free space percentage."""
        pf = (100*self.page_faults)/self.page_ops
        free_space = 0
        for i, level in enumerate(self.free_block_frames):
            free_space += len(level) * (1 << (i+4))     # Adds 4KB * size * number of blocks
        free_space_percent = (100*free_space)/MAIN_MEMORY_SIZE
        return pf, free_space_percent

    
    def __str__(self) -> str:
        vis.draw(self, MIN_BLOCK_SIZE)
        return ""

    def address_to_coord(self, address: int) -> tuple[int, int]:
        """Converts the address to the coordinates in the memory list. Uses cool bit manipulation tricks."""
        # This takes the odd bits out and treats them as a new binary number. Same with even bits.
        address //= MIN_BLOCK_SIZE # turn address into block address (as opposed to page number)
        even_power_bits = 0 # 2^0, 2^2, 2^4, 2^6, ...
        odd_power_bits = 0
        pos = 0
        while address:
            even_power_bits |= ((address & 1) << pos)   # set the relevant bit to match address' bit
            address >>= 1                               # get next address bit

            odd_power_bits |= ((address & 1) << pos)
            address >>= 1
            pos += 1
        return even_power_bits, odd_power_bits


    def rand_blocks_shortcut(self) -> BlockFrame:
        """Shortcut to get a random block frame for testing."""
        not_free = [block_frame for block_frame in self.allocated_block_frames.values() if not block_frame.free]
        return random.choice(not_free)

    def size_to_index(self, size: int) -> int:
        """ Converts the size to the index in the free_blocks list. Same size should give index 0 --> log2(1KB/1KB) = 0"""
        return int(log2((size // MIN_BLOCK_SIZE)))


    def address_or_block(func):
        """Decorator that accepts either an address or block + offset and calls the correct memory function. This reduces duplication for 2 sets of args"""
        def wrapper(self, *args, **kwargs):
            if len(args) == 2:  # Address
                return func(self, *args, **kwargs)
            elif len(args) == 3:    # Block + offset
                block, offset, *args = args
                addr = block.address + offset
                args = (addr, *args)
                return func(self, *args, **kwargs)
            else:
                raise ValueError("Invalid number of arguments")
        return wrapper

    def memoryoperationchecks(func):
        """Decorator for the memory read and write functions. Checks if the block is in memory and moves it if not."""
        def wrapper(self, block: Block, *args, **kwargs):
            #block = self.allocated_block_frames[address]
            if not block.IsInMemory:
                self.page_faults += 1
                self.move_to_memory(block)
            return func(self, block, *args, **kwargs)
        return wrapper

    @memoryoperationchecks
    def write_to_memory(self, block: Block, offset: int, data: list[int]):
        """Writes the data to the memory at the block's address + given offset. If the data provided is too long, an error is raised."""
        assert len(data) <= block.get_size() - offset, "Data too long for block"
        assert offset >= 0 and offset + len(data) <= block.get_size(), "Illegal memory writes are forbidden."
        self.page_ops += 1
        self.memory[ block.address+offset : block.address+offset+len(data)] = data

    @memoryoperationchecks
    def read_from_memory(self, block: Block, offset: int, length: int) -> list[int]:
        """Reads the data from the memory at the block's address + given offset."""
        assert offset >= 0 and offset + length <= block.get_size(), "Illegal memory reads are forbidden"
        self.page_ops += 1
        return self.memory[ block.address+offset : block.address+offset+length]
    
    def move_to_disk(self, block: Block):
        """Moves the block to the disk memory."""
        
        self.disk_lookup[block] = self.next_disk_mem_free
        self.disk_memory[self.next_disk_mem_free : self.next_disk_mem_free+block.get_size()] = self.memory[block.address : block.address+block.get_size()]          # python deals with out-of-bounds slice indexes when writing as appending, fortunately. This is perfect for me.
        self.write_to_memory(block, 0, [None] * block.get_size())   # clear the memory for security
        self.next_disk_mem_free += block.get_size()

        freed_block_frame = block.block_frame
        freed_block_frame.remove_block()
        self.free_block_frames[self.size_to_index(freed_block_frame.get_size())].append(freed_block_frame)
        while freed_block_frame := self.merge_block(freed_block_frame.address):
            ...                                                     # Merge blocks up
        
    
    def move_to_memory(self, block: Block):
        """Moves the block from the disk memory to the main memory."""
        free_block_frame = self.allocate_block_frame(block.get_size(), block)
        self.memory[free_block_frame.address : free_block_frame.address+free_block_frame.get_size()] = self.disk_memory[self.disk_lookup[block] : self.disk_lookup[block]+block.get_size()]
        self.disk_memory[self.disk_lookup[block] : self.disk_lookup[block]+block.get_size()] = [None] * block.get_size()          # clear disk memory for security
        del self.disk_lookup[block]
        # copy the data from the new block to the old block. Ideally we have a seperate object along the lines of page vs page frames.
        self.FIFOBlocks.enqueue(block)


    def allocate_block_frame(self, size: int, block: Block = None) -> BlockFrame:
        """Allocates a block for the requested size."""
        if size > MAIN_MEMORY_SIZE:
            raise InvalidMemoryRequestError(f"Requested size too large {size} > {MAIN_MEMORY_SIZE}")
        
        req_block_frame_size = max(1<<(size-1).bit_length(), MIN_BLOCK_SIZE)                        # Round to next power of 2
        if block_frame_of_right_size := self.free_block_frames[self.size_to_index(req_block_frame_size)]:    
            free_block_frame = block_frame_of_right_size[0]                                         # We know it is not empty
            self.use_block_frame(free_block_frame, block)
            return free_block_frame
        else:                                                                                       # No free block of the right size
            for i in range(self.size_to_index(req_block_frame_size), NUM_BLOCK_SIZES):              # look for next free block
                if self.free_block_frames[i]:
                    free_block_frame = self.free_block_frames[i][0]                                 # It will be removed in the splitting phase
                    while req_block_frame_size != free_block_frame.get_size():                      # Guaranteed to split at least once, loop will break when we get the right size
                        free_block_frame = self.split_block(free_block_frame.address)
                    self.use_block_frame(free_block_frame, block)
                    return free_block_frame
            else:                                                                                   # No free block of any size
                oldest_block = self.fifo_eviction()
                self.move_to_disk(oldest_block)
                self.debug(mem)
                self.debug(f"evict {oldest_block.__str__()}")
                self.debug([block.__str__() for block in self.allocated_block_frames.values()])
                #input()
                return self.allocate_block_frame(size, block)                                       # Try again. Will repeat until enough blocks are evicted

    def deallocate_block(self, block: Block):
        """Deallocates a block by marking it as free and adding it to the free blocks list."""
        b_frame = block.block_frame
        self.write_to_memory(block, 0, [None] * block.get_size())                           # clear the memory for security
        self.free_block_frames[self.size_to_index(b_frame.get_size())].append(b_frame)
        b_frame.remove_block()
        while b_frame := self.merge_block(b_frame.address):
            ...                                                     # placeholder as the block object gets reassigned in the while loop. It ends when 0 is returned (no merge)

    def add_block_frame(self, address: int, size: int) -> BlockFrame:
        """Adds a block by creating the object + adding it to the allocated blocks dictionary."""
        block_frame = BlockFrame(address, size)
        self.allocated_block_frames[address] = block_frame
        return block_frame

    def use_block_frame(self, block_frame: BlockFrame, block: Block):
        """Cleanly removes a block from the free status."""
        block = block or Block(block_frame)     # if block is None, create a new block
        self.free_block_frames[self.size_to_index(block_frame.get_size())].remove(block_frame)
        block_frame.set_block(block)
        self.FIFOBlocks.enqueue(block)

    def split_block(self, addr: int) -> BlockFrame:
        """
            Splits the block frame at the given address.
            Essentially deletes the parent block frame and adds free + allocated child block frames.
            Returns the first child block frame.
        """
        block_frame: BlockFrame = self.allocated_block_frames[addr]
        if block_frame.get_size() == MIN_BLOCK_SIZE:
            raise InvalidMemoryRequestError("Cannot split further")
        if not block_frame.free:
            raise AllocatedBlockFrameError("Block frame is not empty")
        self.free_block_frames[self.size_to_index(block_frame.get_size())].remove(block_frame)    # remove the parent block from free list, as it no longer exists
        
        c1_addr, c2_addr = block_frame.get_children()
        c1 = self.add_block_frame(c1_addr, block_frame.get_size() >> 1)
        c2 = self.add_block_frame(c2_addr, block_frame.get_size() >> 1)

        self.free_block_frames[self.size_to_index(c1.get_size())].append(c1)          # c1 is free until the next split
        self.free_block_frames[self.size_to_index(c2.get_size())].append(c2)          # only c2 is free
    
        return c1
    
    def merge_block(self, addr: int) -> BlockFrame:
        """Merges the block at the given address with its buddy if possible."""
        block_frame : BlockFrame = self.allocated_block_frames[addr]
        buddy_addr = block_frame.get_buddy()
        buddy_frame = self.allocated_block_frames[buddy_addr]
        if buddy_frame.free and (buddy_frame.get_size() == block_frame.get_size()):
            self.free_block_frames[self.size_to_index(buddy_frame.get_size())].remove(buddy_frame)
            self.free_block_frames[self.size_to_index(block_frame.get_size())].remove(block_frame)
            del self.allocated_block_frames[addr]
            del self.allocated_block_frames[buddy_addr]
            
            parent = self.add_block_frame(block_frame.get_parent(), block_frame.get_size() << 1)
            self.free_block_frames[self.size_to_index(parent.get_size())].append(parent)
            self.debug(f"Merge: {block_frame.__str__()}, {buddy_frame.__str__()}, {parent.__str__()}")
            return parent
        return 0    # no merge
    
    def fifo_eviction(self) -> Block:
        """Evicts the oldest block in the memory."""
        while True:
            oldest_block = self.FIFOBlocks.dequeue()
            if oldest_block.IsInMemory:
                break                               # ensures it exists still
        # Second chance algorithm or clock algorithm possible here. Currently just a wrapper function. For futureproofing
        return oldest_block


import random

if __name__ == "__main__":
    mem = Memory()
    programs = []
    for i in range(60):
        _block = mem.allocate_block_frame(1 << random.randint(1, 10)).block
        programs.append(_block)
        mem.debug(mem)
        mem.debug([b.block.__str__() for b in mem.allocated_block_frames.values()])
        #input()

    for i in range(100): #  simulate 10000 memory operations
        _block = random.choice(programs)
        if random.random() < 0.5:
            mem.write_to_memory(_block, 0, [1])  # everything is treated block-by-block, so no point writing more than 1 element
            mode = "WRITE"
        else:
            d = mem.read_from_memory(_block, 0, 16)
            mode = "READ"
        mem.debug(mem)
        mem.debug([b.block.__str__() for b in mem.allocated_block_frames.values()])
        with open("mem.txt", "a") as f:
            pf, free = mem.stats()
            f.write(f"{mode}\tPage fault rate: {pf:.2f}%\t\tFree memory: {free:.2f}%\n")
        #input("Enter: ")
    pf, free_m = mem.stats()
    print(f"Page fault rate: {pf}%")
    print(f"Free memory: {free_m}%")

