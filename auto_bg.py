'''
This code will automatically annotate a "background" bouding box in all the images in the selected dataset. 
'''


import os
import csv
import random

WIDTH = 640
HEIGHT = 480
MIN_RECT_SIZE = 50
BASE_DIR = 'data_bg'

def rects_intersect(r1, r2):
    """
    Check if two rectangles intersect.
    Each rectangle is a tuple: (x, y, width, height),
    where (x, y) is the top-left corner.
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    # Compute edges
    r1_right, r1_bottom = x1 + w1, y1 + h1
    r2_right, r2_bottom = x2 + w2, y2 + h2

    # Check for separation along x or y axis
    if r1_right <= x2 or r2_right <= x1:
        return False
    if r1_bottom <= y2 or r2_bottom <= y1:
        return False

    return True

def get_non_intersecting_rect(target_rects):
    '''
    When given a set of rectangles, return a random rectangle that does not intersect with any of the rectangles. 
    '''

    # Horrifyingly lazy algorithm. Please forgive me lol. 

    while True:

        # Random top left coordinate
        start_x = random.randint(0, WIDTH - MIN_RECT_SIZE - 1)
        start_y = random.randint(0, HEIGHT - MIN_RECT_SIZE - 1)

        # Biggest thing we could fit in this space
        max_width = WIDTH - start_x
        max_height = HEIGHT - start_y

        # To ensure we don't run over the edge
        max_size = min(max_width, max_height)

        size = random.randint(MIN_RECT_SIZE, max_size)

        rect = (start_x, start_y, size, size) # yeah its a square. shhhhhh

        for target_rect in target_rects:
            if rects_intersect(target_rect, rect):
                continue # This rect intersected with one of the specified rects. Try again. 

        return rect

def add_bg_class(path):
    '''
    Add a background class to the desired CSV path. If a bg class is already added, do nothing. 
    '''

    rects = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["width"])
            h = float(row["height"])
            c = row["class"]

            if c == 'background':
                return # Background already added. 

            rects.append((x, y, w, h))
    
    new_rect = get_non_intersecting_rect(rects)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_rect + ('background',))


if __name__ == '__main__':

    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if f.lower().split('.')[-1] not in ['jpg','jpeg','png']:
                continue

            img_path = os.path.join(root, f)
            csv_path = img_path.rsplit('.',1)[0] + '.csv'
            if not os.path.exists(csv_path): continue

            add_bg_class(csv_path)