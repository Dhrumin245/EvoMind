import pygame

class ArenaRenderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

    def render(self, env):
        self.screen.fill((0, 0, 0))

        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (int(env.agent_x), int(env.agent_y)),
            10
        )

        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            (int(env.food_x), int(env.food_y)),
            8
        )

        pygame.display.flip()
        self.clock.tick(60)
