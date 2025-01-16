import pygame
import time
import random
import numpy as np
from model import QNetwork

pygame.init()

WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20
SNAKE_SPEED = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)
top_score = 0

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake RL")

clock = pygame.time.Clock()

font = pygame.font.SysFont("bahnschrift", 25)

def display_score(score):
    value = font.render("Current score: " + str(score), True, WHITE)
    screen.blit(value, [0, 0])

def draw_snake(block_size, snake_list):
    for block in snake_list:
        pygame.draw.rect(screen, GREEN, [block[0], block[1], block_size, block_size])

def collect_state(head, snake_list, food_x, food_y, current_direction):
    def is_danger(ahead_x, ahead_y):
        # Check if position is outside the boundaries or collides with the snake's body
        return (
            ahead_x >= WIDTH or ahead_x < 0 or
            ahead_y >= HEIGHT or ahead_y < 0 or
            [ahead_x, ahead_y] in snake_list
        )

    # Determine danger in each direction
    danger_straight = is_danger(
        head[0] + (current_direction == 'RIGHT') * BLOCK_SIZE - (current_direction == 'LEFT') * BLOCK_SIZE,
        head[1] + (current_direction == 'DOWN') * BLOCK_SIZE - (current_direction == 'UP') * BLOCK_SIZE
    )
    danger_right = is_danger(
        head[0] + (current_direction == 'UP') * BLOCK_SIZE - (current_direction == 'DOWN') * BLOCK_SIZE,
        head[1] + (current_direction == 'RIGHT') * BLOCK_SIZE - (current_direction == 'LEFT') * BLOCK_SIZE
    )
    danger_left = is_danger(
        head[0] + (current_direction == 'DOWN') * BLOCK_SIZE - (current_direction == 'UP') * BLOCK_SIZE,
        head[1] + (current_direction == 'LEFT') * BLOCK_SIZE - (current_direction == 'RIGHT') * BLOCK_SIZE
    )

    # One-hot encoding of the current direction
    dir_up = current_direction == 'UP'
    dir_down = current_direction == 'DOWN'
    dir_left = current_direction == 'LEFT'
    dir_right = current_direction == 'RIGHT'

    # Relative position of the food
    food_left = food_x < head[0]
    food_right = food_x > head[0]
    food_up = food_y < head[1]
    food_down = food_y > head[1]

    state = [
        int(danger_straight),
        int(danger_right),
        int(danger_left),
        int(dir_up),
        int(dir_down),
        int(dir_left),
        int(dir_right),
        int(food_left),
        int(food_right),
        int(food_up),
        int(food_down)
    ]

    return np.array(state)

def get_reward(snake_head, prev_distance, food_x, food_y, new_distance, game_close):
    if game_close:
        return -100  # Penalty for dying
    elif snake_head[0] == food_x and snake_head[1] == food_y:
        return 100  # Reward for eating food
    else:
        # Calculate new distance to food
        if new_distance < prev_distance:
            return 4  # Reward for moving closer
        else:
            return -2  # Penalty for moving away
        
def set_top_score(score):
    top_score = score
        

ACTIONS = ['STRAIGHT', 'LEFT', 'RIGHT']
ACTIONS_KEY = {'STRAIGHT': 0, 'LEFT': 1, 'RIGHT': 2}

# Define the RL Agent
agent = QNetwork(state_size=11, action_size=3)
episode_rewards = []

def turn_left(direction):
    turn_map = {'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT', 'RIGHT': 'UP'}
    return turn_map[direction]

def turn_right(direction):
    turn_map = {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP'}
    return turn_map[direction]

def game_loop(episode):

    batch_size = 32
    episode_reward = 0
    episode += 1
    step_counter = 0
    training_interval = 100

    game_over = False
    game_close = False

    x = WIDTH / 2
    y = HEIGHT / 2

    current_direction = 'RIGHT'  # Initial direction
    x_change = BLOCK_SIZE
    y_change = 0

    snake_list = []
    snake_len = 1

    food_x = round(random.randrange(0, WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
    food_y = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE

    print(f"Episode: {episode}")
    while not game_over:
        step_counter += 1

        prev_distance = abs(x - food_x) + abs(y - food_y)

        snake_head = [x, y]

        snake_list.append(snake_head)
        if len(snake_list) > snake_len:
            del snake_list[0]

        state = collect_state(snake_head, snake_list, food_x, food_y, current_direction)
        action = agent.action(state)

        # Process action
        if action == 0:  # Straight
            new_direction = current_direction
        elif action == 1:  # Left
            new_direction = turn_left(current_direction)
        elif action == 2:  # Right
            new_direction = turn_right(current_direction)

        current_direction = new_direction

        # Update x_change and y_change based on current_direction
        if current_direction == 'UP':
            x_change = 0
            y_change = -BLOCK_SIZE
        elif current_direction == 'DOWN':
            x_change = 0
            y_change = BLOCK_SIZE
        elif current_direction == 'LEFT':
            x_change = -BLOCK_SIZE
            y_change = 0
        elif current_direction == 'RIGHT':
            x_change = BLOCK_SIZE
            y_change = 0

        # Handle events without user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Move the snake
        x += x_change
        y += y_change

        new_distance = abs(x - food_x) + abs(y - food_y)

        # Check for collisions
        if x >= WIDTH or x < 0 or y >= HEIGHT or y < 0:
            game_close = True

        screen.fill(BLACK)

        # Draw food
        pygame.draw.rect(screen, RED, [food_x, food_y, BLOCK_SIZE, BLOCK_SIZE])

        # Check if snake hits itself
        for block in snake_list[:-1]:
            if block == snake_head:
                game_close = True

        draw_snake(BLOCK_SIZE, snake_list)
        display_score(snake_len - 1)

        # Refresh the screen
        pygame.display.update()

        # Update the reward
        reward = get_reward(snake_head, prev_distance, food_x, food_y, new_distance, game_close)
        episode_reward += reward

        next_state = collect_state(snake_head, snake_list, food_x, food_y, current_direction)
        agent.remember(state, action, reward, next_state, done=game_close)

        # Train the agent
        if step_counter % training_interval == 0 and len(agent.memory) > batch_size:
            agent.replay(batch_size)
        

        if game_close:
            episode_rewards.append(episode_reward)
            print(f"Episode Reward: {episode_reward}")
            print(f"Epsilon: {agent.epsilon}")
            game_over = True  # End the game

        # If snake eats food
        if x == food_x and y == food_y:
            while True:
                food_x = round(random.randrange(0, WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
                food_y = round(random.randrange(0, HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
                if [food_x, food_y] not in snake_list:
                    break
            snake_len += 1

        clock.tick(SNAKE_SPEED)

def main():
    episode = 0
    while True:
        game_loop(episode)
        episode += 1

if __name__ == "__main__":
    main()
