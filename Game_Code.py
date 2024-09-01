import pygame
import random
import sys
import cv2
import mediapipe as mp
import numpy as np
import win32api
from pygame.locals import *
import pyautogui as pg

pygame.init()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
CYAN = (0,255,255)
MAGENTA = (255,0,255)

WINDOWHEIGHT = 1200
WINDOWWIDTH = 1920

FONT = pygame.font.SysFont(None, 48)
mainClock = pygame.time.Clock()

score = 0
highscore = 0

windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
pygame.display.set_caption("Nose Tracking Aim Trainer")

vid = cv2.VideoCapture(0)

nose_image = cv2.imread("Nose.png", cv2.IMREAD_UNCHANGED)
targetImage = pygame.image.load("TNT.png")
targetImage = pygame.transform.scale(targetImage, (75, 75))

explosionImage = pygame.image.load("boom.png")
explosionImage = pygame.transform.scale(explosionImage, (100, 100))

def terminate():
    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

def drawText(text, surface, x, y, font=FONT, color=CYAN):
    textObject = font.render(text, 1, color)
    textRect = textObject.get_rect()
    textRect.topleft = (x, y)
    surface.blit(textObject, textRect)

def gameOver(score):
    global highscore
    pygame.mouse.set_visible(True)
    windowSurface.fill(BLACK)
    drawText("GAME OVER", windowSurface, 765, 350, pygame.font.SysFont(None, 72, True))
    drawText(f"Score: {score}", windowSurface, 800, 450)
    drawText(f"Highest score: {highscore}", windowSurface, 800, 500)
    drawText("Play Again (SPACE) or Quit (ESC)", windowSurface, 650, 600)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    terminate() 
                elif event.key == pygame.K_SPACE:
                    cv2.destroyAllWindows()
                    return

def nose_tracking():
    success, img = vid.read()
    if not success:
        print("Failed to capture image")
        return None, None
    
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    nose_pos = None
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = img.shape
            nose_tip = face_landmarks.landmark[1]
            nose_x = int(nose_tip.x * w)
            nose_y = int(nose_tip.y * h)
 
            screen_x = np.interp(nose_x, [0, w], [0, WINDOWHEIGHT]) 
            screen_y = np.interp(nose_y, [0, h], [0, WINDOWWIDTH]) 

            win32api.SetCursorPos((int(screen_y), int(screen_x)))
            
            nose_image_resized = cv2.resize(nose_image, (150, 150))
            nose_alpha = nose_image_resized[:, :, 3] / 255.0
            img_alpha = 1.0 - nose_alpha

            y1, y2 = int(nose_y - 75), int(nose_y + 75)
            x1, x2 = int(nose_x - 55), int(nose_x + 95)

            if y1 >= 0 and y2 < h and x1 >= 0 and x2 < w:
                for c in range(3):
                    img[y1:y2, x1:x2, c] = (nose_alpha * nose_image_resized[:, :, c] +
                                             img_alpha * img[y1:y2, x1:x2, c])

            nose_pos = (int(screen_y), int(screen_x))

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), nose_pos

def main_game_loop():
    global score, highscore
    pygame.mouse.set_visible(True)
    enemies = []
    amountOfEnemies = 0
    score = 0
    FPS = 30
    hitShots = 0
    totalShots = 0
    STARTINGTIME = 31
    start_time = pygame.time.get_ticks()
    last_update_time = start_time

    while STARTINGTIME > 1:
        current_time = pygame.time.get_ticks()
        delta_time = (current_time - last_update_time) / 1000
        STARTINGTIME -= delta_time
        last_update_time = current_time
        if STARTINGTIME <= 1:
            STARTINGTIME = 1

        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            if event.type == KEYUP and event.key == K_ESCAPE:
                terminate()
            if event.type == MOUSEBUTTONDOWN:
                totalShots += 1
                for enemy in enemies[:]:
                    if enemy.collidepoint(event.pos):
                        enemies.remove(enemy)
                        amountOfEnemies -= 1
                        score += 1
                        hitShots += 1
                        break

        if amountOfEnemies < 5:
            new_enemy = pygame.Rect(
                random.randint(50, WINDOWWIDTH - 50),
                random.randint(50, WINDOWHEIGHT - 50),
                50, 50
            )
            enemies.append(new_enemy)
            amountOfEnemies += 1

        windowSurface.fill(BLACK)
        
        nose_img, nose_pos = nose_tracking()
        if nose_img is not None:
            nose_surf = pygame.surfarray.make_surface(nose_img)
            nose_surf = pygame.transform.scale(nose_surf, (WINDOWWIDTH, WINDOWHEIGHT))
            windowSurface.blit(nose_surf, (0, 0))

        for enemy in enemies[:]:
            if nose_pos and enemy.collidepoint(nose_pos):
                windowSurface.blit(explosionImage, enemy)
                pygame.display.update()
                pygame.time.delay(100)
                enemies.remove(enemy)
                amountOfEnemies -= 1
                score += 1
            else:
                windowSurface.blit(targetImage, enemy)

        drawText(f"Time: {int(STARTINGTIME)}", windowSurface, 10, 10, color=WHITE)
        drawText(f"Score: {score}", windowSurface, 10, 50, color=WHITE)

        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.line(windowSurface, GREEN, (mouse_pos[0] - 10, mouse_pos[1]), (mouse_pos[0] + 10, mouse_pos[1]), 2)
        pygame.draw.line(windowSurface, GREEN, (mouse_pos[0], mouse_pos[1] - 10), (mouse_pos[0], mouse_pos[1] + 10), 2)

        pygame.display.update()
        mainClock.tick(FPS)

    if score > highscore:
        highscore = score

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            main_game_loop()

    gameOver(score)

while True:
    main_game_loop()
    cv2.destroyAllWindows()