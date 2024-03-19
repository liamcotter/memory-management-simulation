# visualistion
from time import sleep
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"   # hide the pygame support prompt, which is shouldn't be in the library in the first place...
import pygame
from math import log2, sqrt
pygame.init()
screen_size = 750
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Memory Visualisation")

# Future things to do:
# Change the way memory is written to (use address only, ideally)
# Add more stringent checks for security
# Use better memory management algorithms
# Intergate with the process manager + more?

# All sizes are in KB

from tools import Queue
MAIN_MEMORY_SIZE = 1 << 12  # 4096 KB
PAGE_SIZE = 1 << 2  # 4 KB
MIN_BLOCK_SIZE = PAGE_SIZE << 2 # 16 KB
NUM_BLOCK_SIZES = int(log2(MAIN_MEMORY_SIZE // MIN_BLOCK_SIZE)) + 1 # must include biggest and smallest size, so +1

usable_space = screen_size * 1
vis_size = sqrt(MAIN_MEMORY_SIZE // MIN_BLOCK_SIZE) # max number of blocks, sqrt to get the sides


class Block:
    def __init__(self, address: int, size: int):
        assert address % size == 0, "Block out of alignment"
        self.address = address
        self.size = size
        self.free = True
        self.IsInMemory = True
        self.exists = True # might be added to queue but subsequently removed. Bit set for efficiency.

    def __str__(self) -> str:
        return f"Address: {self.address}, Size: {self.size}, Free: {self.free}, IsInMemory: {self.IsInMemory}"
    
    def get_size(self) -> int:
        """Gets the size of the block"""
        return self.size
    
    def get_buddy(self) -> int:
        """Gets the address of the buddy block"""
        return self.address ^ self.size  # address XOR size = buddy
    
    def get_parent(self) -> int:
        """Gets the address of the parent block"""
        return self.address & ~((self.size << 1) - 1) # address AND NOT (2*size-1) = parent
    
    def get_children(self) -> list[int]:
        """Gets the address of the two child blocks upon splitting the block."""
        return [self.address, self.address + (self.size >> 1)] # address, address + size = children

class Memory:
    def __init__(self):
        self.memory = [None] * MAIN_MEMORY_SIZE
        self.disk_memory = [None] * MAIN_MEMORY_SIZE    # should be much larger, but no point simulating too much for proof-of-concept
        self.disk_lookup = {}                           # Should be handled as virtual memory, but it is too long to implement in a lab, so we just store the addresses for the moved blocks.
        self.next_disk_mem_free = 0                     # Next free address in the disk. Very inefficient, probably should be looping, but it is a quick solution to evicted blocks. Ideally a similar memory manager, but simpler should be used.
        self.FIFOBlocks = Queue()                       # FIFO but with entire blocks instead of pages
        self.free_blocks = [[] for _ in range(NUM_BLOCK_SIZES)]
        self.allocated_blocks = {}

        init_block = self.add_block(0, MAIN_MEMORY_SIZE)
        self.free_blocks[self.size_to_index(init_block.get_size())].append(init_block)


        # For stats
        self.page_faults = 0
        self.page_ops = 0

        self.print_debug = False        # Quick enabling of debug

    def debug(self, item):
        """Prints the debug message if the debug flag is set."""
        if self.print_debug and type(item) == Memory:
            print(item.__str__())
    
    def stats(self) -> tuple[float, float]:
        """Returns the stats of the memory. Returns the page fault rate and the free space percentage."""
        pf = (100*self.page_faults)/self.page_ops
        free_space = 0
        for i, level in enumerate(self.free_blocks):
            free_space += len(level) * (1 << (i+4))     # Adds 4KB * size * number of blocks
        free_space_percent = (100*free_space)/MAIN_MEMORY_SIZE
        return pf, free_space_percent

    
    def __str__(self) -> str:
        per_block_space = usable_space / vis_size
        screen.fill((255, 255, 255))
        for block in self.allocated_blocks.values():
            addr = block.address
            size = block.size
            x, y = self.address_to_coord(addr)
            real_x, real_y = x * per_block_space, y * per_block_space

            blocks_size = size // MIN_BLOCK_SIZE                        # units as multiples of block minimum size
            if log2(blocks_size) % 2 != 0:  # isn't square
                blocks_size = int(sqrt(blocks_size//2))
                x_size = 2 * per_block_space * blocks_size
                y_size = per_block_space * blocks_size
            else:
                blocks_size = int(sqrt(blocks_size))                    # width = sqrt(area)
                x_size = per_block_space * blocks_size
                y_size = per_block_space * blocks_size
            
            # Filled Square
            if block.free:
                pygame.draw.rect(screen, (0, 255, 0), (real_x, real_y, x_size, y_size))
            else:
                pygame.draw.rect(screen, (255, 0, 0), (real_x, real_y, x_size, y_size))
            # Border
            pygame.draw.rect(screen, (0, 0, 0), (real_x, real_y, x_size, y_size), width=1)
            
        pygame.display.flip()
        sleep(0.1)
        #pygame.quit()
        """while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return """""
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
        
    def size_to_index(self, size: int) -> int:
        """ Converts the size to the index in the free_blocks list. Same size should give index 0 --> log2(1KB/1KB) = 0"""
        return int(log2((size // MIN_BLOCK_SIZE)))

    def memoryoperationchecks(func):
        """Decorator for the memory read and write functions. Checks if the block is in memory and moves it if not."""
        def wrapper(self, block: Block, *args, **kwargs):
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
        self.disk_memory[self.next_disk_mem_free : self.next_disk_mem_free+block.size] = self.memory[block.address : block.address+block.size]          # python deals with out-of-bounds slice indexes when writing as appending, fortunately. This is perfect for me.
        self.write_to_memory(block, 0, [None] * block.get_size())   # clear the memory for security
        block.IsInMemory = False
        self.next_disk_mem_free += block.size

        del self.allocated_blocks[block.address]
        empty_block = self.add_block(block.address, block.size)
        self.free_blocks[self.size_to_index(empty_block.get_size())].append(empty_block)
        self.allocated_blocks[block.address] = empty_block

        while block := self.merge_block(block.address):
            ...                                                     # Merge blocks up
    
    def move_to_memory(self, block: Block):
        """Moves the block from the disk memory to the main memory."""
        free_block = self.allocate_block(block.get_size())
        self.memory[free_block.address : free_block.address+free_block.size] = self.disk_memory[self.disk_lookup[block] : self.disk_lookup[block]+block.size]
        self.disk_memory[self.disk_lookup[block] : self.disk_lookup[block]+block.size] = [None] * block.get_size()
        del self.disk_lookup[block]
        # copy the data from the new block to the old block. Ideally we have a seperate object along the lines of page vs page frames.
        free_block.exists = False
        block.IsInMemory = True
        block.address = free_block.address
        self.allocated_blocks[free_block.address] = block
        self.FIFOBlocks.enqueue(block)


    def allocate_block(self, size: int) -> Block:
        """Allocates a block for the requested size."""
        assert size <= MAIN_MEMORY_SIZE, "Requested size too large"
        req_block_size = max(1<<(size-1).bit_length(), MIN_BLOCK_SIZE)                      # Round to next power of 2
        if blocks_of_right_size := self.free_blocks[self.size_to_index(req_block_size)]:    
            free_block = blocks_of_right_size[0]                                            # We know it is not empty
            self.use_block(free_block)
            return free_block
        else:                                                                               # No free block of the right size
            for i in range(self.size_to_index(req_block_size), NUM_BLOCK_SIZES):            # look for next free block
                if self.free_blocks[i]:
                    free_block = self.free_blocks[i][0]                                     # It will be removed in the splitting phase
                    while req_block_size != free_block.get_size():                          # Guaranteed to split at least once
                        free_block = self.split_block(free_block.address)
                    self.use_block(free_block)
                    return free_block             
            else:
                oldest_block = self.fifo_eviction()
                self.move_to_disk(oldest_block)
                self.debug(mem)
                self.debug(f"evict {oldest_block.__str__()}")
                self.debug([block.__str__() for block in self.allocated_blocks.values()])
                #input()
                return self.allocate_block(size)                                            # Try again. Will repeat until enough blocks are evicted

    def deallocate_block(self, block: Block):
        """Deallocates a block by marking it as free and adding it to the free blocks list."""
        block.free = True
        self.write_to_memory(block, 0, [None] * block.get_size())                           # clear the memory for security
        self.free_blocks[self.size_to_index(block.get_size())].append(block)
        self.allocated_blocks[block.address] = block
        while block := self.merge_block(block.address):
            ...                                                     # placeholder as the block object gets reassigned in the while loop. It ends when 0 is returned (no merge)

    def add_block(self, address: int, size: int) -> Block:
        """Adds a block by creating the object + adding it to the allocated blocks dictionary."""
        block = Block(address, size)
        self.allocated_blocks[address] = block
        return block

    def use_block(self, block: Block):
        """Cleanly removes a block from the free status."""
        self.free_blocks[self.size_to_index(block.get_size())].remove(block)
        block.free = False
        self.FIFOBlocks.enqueue(block)

    def split_block(self, addr: int) -> Block:
        """
            Splits the block at the given address.
            Essentially deletes the parent block and adds free + allocated child blocks.
            Returns the first child block.
        """
        block: Block = self.allocated_blocks[addr]
        if block.get_size() == MIN_BLOCK_SIZE:
            raise ValueError("Cannot split further")
        
        self.free_blocks[self.size_to_index(block.get_size())].remove(block)    # remove the parent block from free list, as it no longer exists
        
        c1_addr, c2_addr = block.get_children()
        c1 = self.add_block(c1_addr, block.get_size() >> 1)
        c2 = self.add_block(c2_addr, block.get_size() >> 1)

        self.free_blocks[self.size_to_index(c1.get_size())].append(c1)          # c1 is free until the next split
        self.free_blocks[self.size_to_index(c2.get_size())].append(c2)          # only c2 is free
        

        return c1
    
    def merge_block(self, addr: int) -> Block:
        """Merges the block at the given address with its buddy if possible."""
        block: Block = self.allocated_blocks[addr]
        buddy_addr = block.get_buddy()
        if buddy_addr in self.allocated_blocks and self.allocated_blocks[buddy_addr].free and self.allocated_blocks[buddy_addr].size == block.size:
            buddy = self.allocated_blocks[buddy_addr]
            self.free_blocks[self.size_to_index(buddy.get_size())].remove(buddy)
            self.free_blocks[self.size_to_index(block.get_size())].remove(block)
            del self.allocated_blocks[addr]
            del self.allocated_blocks[buddy_addr]
            
            parent = self.add_block(block.get_parent(), block.get_size() << 1)
            self.free_blocks[self.size_to_index(parent.get_size())].append(parent)
            self.debug(f"Merge: {block.__str__()}, {buddy.__str__()}, {parent.__str__()}")
            return parent
        return 0    # no merge
    
    def fifo_eviction(self) -> Block:
        """Evicts the oldest block in the memory."""
        while True:
            oldest_block = self.FIFOBlocks.dequeue()
            if oldest_block.exists:
                break                               # ensures it exists still
        # Second chance algorithm or clock algorithm possible here. Currently just a wrapper function. For futureproofing
        return oldest_block


import random

if __name__ == "__main__":
    mem = Memory()
    programs = []
    for i in range(60):
        block = mem.allocate_block(1 << random.randint(1, 11))
        programs.append(block)
        mem.debug(mem)
        mem.debug([block.__str__() for block in mem.allocated_blocks.values()])
        #input()

    for i in range(100): #  simulate 10000 memory operations
        block = random.choice(programs)
        if random.random() < 0.5:
            mem.write_to_memory(block, 0, [1])  # everything is treated block-by-block, so no point writing more than 1 element
            mode = "WRITE"
        else:
            mem.read_from_memory(block, 0, 16)
            mode = "READ"
        mem.debug(mem)
        mem.debug([block.__str__() for block in mem.allocated_blocks.values()])
        with open("mem.txt", "a") as f:
            pf, free = mem.stats()
            f.write(f"{mode}\tPage fault rate: {pf:.2f}%\t\tFree memory: {free:.2f}%\n")
        #input("Enter: ")
    #print(mem)
    pf, free_m = mem.stats()
    print(f"Page fault rate: {pf}%")
    print(f"Free memory: {free_m}%")


    # b = mem.allocate_block(MAIN_MEMORY_SIZE)
    # mem.write_to_memory(b, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print("MEM: ", mem.memory)
    # print("DISK: ", mem.disk_memory)
    # c = mem.allocate_block(1)
    # mem.write_to_memory(c, 0, [1]*c.get_size())
    # print("MEM: ", mem.memory)
    # print("DISK: ", mem.disk_memory)
    # d = mem.allocate_block(MAIN_MEMORY_SIZE)
    # mem.write_to_memory(d, 0, [2]*d.get_size())
    # print("MEM: ", mem.memory)
    # print("DISK: ", mem.disk_memory)
    # print(mem.read_from_memory(c, 0, 1))
    # print("MEM: ", mem.memory)
    # print("DISK: ", mem.disk_memory)
    # e = mem.allocate_block(1)
    # mem.write_to_memory(e, 0, [3]*e.get_size())
    # bfull = mem.allocate_block(4096)
    # mem.deallocate_block(bfull)
    # b16 = mem.allocate_block(16)
    # b127 = mem.allocate_block(127)
    # b513 = mem.allocate_block(1024)
    # mem.allocate_block(255)
    # mem.allocate_block(130)
    # mem.allocate_block(55)
    # mem.allocate_block(99)
    # mem.allocate_block(129)
    # mem.deallocate_block(b127)
    # mem.allocate_block(90)
    # mem.deallocate_block(b513)
    # mem.allocate_block(257)
    # mem.allocate_block(154)
    # mem.allocate_block(103)
    # mem.allocate_block(48)
    # mem.allocate_block(3)
    # mem.deallocate_block(b16)
    #print(mem)
