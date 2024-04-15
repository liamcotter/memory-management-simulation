from time import sleep
import pygame
from math import log2, sqrt


class Visualiser(object):
    def __init__(self, max_blocks):
        """Initialise the visualisation environment."""
        pygame.init()
        pygame.display.set_caption("Memory Visualisation")
        screen_size = 750
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.usable_space = screen_size * 1
        self.vis_size = sqrt(max_blocks) # max number of blocks, sqrt to get the sides
        self.per_block_space = self.usable_space / self.vis_size
    
    def draw(self, parent, MIN_BLOCK_SIZE):
        self.screen.fill((255, 255, 255))
        for block in parent.allocated_blocks.values():
            addr = block.address
            size = block.size
            x, y = parent.address_to_coord(addr)
            real_x, real_y = x * self.per_block_space, y * self.per_block_space

            blocks_size = size // MIN_BLOCK_SIZE                        # units as multiples of block minimum size
            if log2(blocks_size) % 2 != 0:  # isn't square
                blocks_size = int(sqrt(blocks_size//2))
                x_size = 2 * self.per_block_space * blocks_size
                y_size = self.per_block_space * blocks_size
            else:
                blocks_size = int(sqrt(blocks_size))                    # width = sqrt(area)
                x_size = self.per_block_space * blocks_size
                y_size = self.per_block_space * blocks_size
            
            # Filled Square
            if block.free:
                pygame.draw.rect(self.screen, (0, 255, 0), (real_x, real_y, x_size, y_size))
            else:
                pygame.draw.rect(self.screen, (255, 0, 0), (real_x, real_y, x_size, y_size))
            # Border
            pygame.draw.rect(self.screen, (0, 0, 0), (real_x, real_y, x_size, y_size), width=1)
            
        pygame.display.flip()
        sleep(0.1)
        # pygame.quit()
        return ""
