import pygame

pygame.init()

gameDisplay = pygame.display.set_mode((500, 500))
pygame.display.set_caption('A bit snakey')

black = (0, 0, 0)
white = (255, 255, 255)

clock = pygame.time.Clock()

crashed = False
snakeImg = pygame.image.load('snake.jpg')


def snake(x, y):
    gameDisplay.blit(snakeImg, (x, y))


x = (500 * 0.10)
y = (500 * 0.20)

while not crashed:

    for event in pygame.event.get():
        print(event)
        if event.type == pygame.QUIT:
            crashed = True
    gameDisplay.fill(black)
    snake(x, y)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
