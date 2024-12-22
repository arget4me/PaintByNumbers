
# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from functools import partial
from random import choice


def get_closest(value : float, values : list[float], maxValue : float = 10000000.0) -> float:
    outValue = value
    dist = maxValue

    for v in values:
        newDist = abs(v - value)
        if newDist < dist:
            dist = newDist
            outValue = v
        else:
            break
    return outValue

def get_closest_floor(value : float, values : list[float]) -> float:
    outValue = 0
    for v in values:
        if value < v:
            break
        outValue = v
    return outValue

def apply_canny_edge_detection(image, threshold1, threshold2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    result = image.copy()
    result[edges > 0] = [0, 0, 0]
    result[edges <= 0] = [255, 255, 255]

    return edges, result

def overlay_edges_on_image(image, threshold1, threshold2):
    edges, _ = apply_canny_edge_detection(image, threshold1, threshold2)
    overlay = image.copy()
    
    # Set the edge pixels to black on the overlay
    overlay[edges > 0] = [0, 0, 0]
    
    return overlay

def gaussian_blur_selected_channels(image, ksize, sigmaX, channels_to_blur):
    r, g, b = cv2.split(image)

    channels = [r, g, b]

    for i in channels_to_blur:
        channels[i] = cv2.GaussianBlur(channels[i], (ksize, ksize), sigmaX)

    return cv2.merge(channels)

hues = range(0, 256, 256//11)
saturations = range(0, 256, 256//2)
values = range(0, 256, 256//3)

def pixel_process(pixelHSV) -> list:
    pixelHSV[0] = get_closest(pixelHSV[0], hues)
    pixelHSV[1] = get_closest(pixelHSV[1], saturations)
    pixelHSV[2] = get_closest(pixelHSV[2], values)
    return pixelHSV

def pixel_process_simpel(pixelHSV) -> list:
    pixelHSV[0] = get_closest(pixelHSV[0], hues)
    pixelHSV[1] = 255
    return pixelHSV

def process_row(yPos, srcImage):
    row = srcImage[yPos]
    return np.array([pixel_process(pixel) for pixel in row])

def parallel_process_all_pixels(srcImage, axis):
    imageHeight, imageWidth, _ = srcImage.shape
    newImg = np.zeros_like(srcImage)

    process_row_partial = partial(process_row, srcImage=srcImage)

    with Pool() as pool:
        result = pool.map(process_row_partial, range(imageHeight))

    for yPos in range(imageHeight):
        newImg[yPos] = result[yPos]

    axis.imshow(cv2.cvtColor(newImg, cv2.COLOR_HSV2RGB))
    plt.draw()
    plt.pause(0.001)

    return newImg




def get_unique_colors(image):
    pixels = image.reshape(-1, image.shape[2])
    unique_colors = np.unique(pixels, axis=0)
    return unique_colors

# def find_small_segments(image, unique_colors, min_area):
#     small_segments = {}
    
#     for color in unique_colors:
#         mask = np.all(image == color, axis=-1)
#         area = np.sum(mask)
#         if area < min_area:
#             small_segments[tuple(color)] = area
    
#     return small_segments

# def find_neighbors(image, color):
#     neighbors = set()
#     rows, cols, _ = image.shape
#     mask = np.all(image == color, axis=-1)
    
#     for r in range(rows):
#         for c in range(cols):
#             if mask[r, c]:
#                 # Check 8-connected neighborhood
#                 for dr in [-1, 0, 1]:
#                     for dc in [-1, 0, 1]:
#                         if dr == 0 and dc == 0:
#                             continue
#                         nr, nc = r + dr, c + dc
#                         if 0 <= nr < rows and 0 <= nc < cols:
#                             neighbor_color = tuple(image[nr, nc])
#                             if neighbor_color != tuple(color):
#                                 neighbors.add(neighbor_color)
    
#     return list(neighbors)

# def merge_small_segments(image, small_segments):
#     for color in small_segments.keys():
#         neighbors = find_neighbors(image, color)
#         if neighbors:
#             # Choose a random neighbor color
#             new_color = choice(neighbors)
#             mask = np.all(image == color, axis=-1)
#             image[mask] = new_color
    
#     return image




def find_largest_touching_neighbor(image, color, submask):
    neighbors = {}
    rows, cols, _ = image.shape
    mask = np.all(image == color, axis=-1)
    touched = np.zeros((rows, cols), dtype=bool)
    
    # Get the bounding box of the submask to limit search area
    bbox = cv2.boundingRect(submask.astype(np.uint8))
    x, y, w, h = bbox
    submask_region = submask[y:y+h, x:x+w]
    
    for r in range(y, y+h):
        for c in range(x, x+w):
            if submask_region[r-y, c-x]:
                # Check 8-connected neighborhood
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not touched[nr, nc]:
                                neighbor_color = tuple(image[nr, nc])
                                if neighbor_color != tuple(color):
                                    if neighbor_color not in neighbors:
                                        neighbors[neighbor_color] = np.sum(np.all(image == neighbor_color, axis=-1))
                                    touched[nr, nc] = True
                                    
    if not neighbors:
        return None

    # Find the largest neighbor
    largest_neighbor = max(neighbors, key=neighbors.get)
    return largest_neighbor

def find_largest_touching_segment(image, segment, segments):
    neighbors = {}
    rows, cols, _ = image.shape
    color = segment[0]
    submask = segment[1]
    touched = np.zeros((rows, cols), dtype=bool)
    
    # Get the bounding box of the submask to limit search area
    bbox = cv2.boundingRect(submask.astype(np.uint8))
    x, y, w, h = bbox
    submask_region = submask[y:y+h, x:x+w]
    
    for r in range(y, y+h):
        for c in range(x, x+w):
            if submask_region[r-y, c-x]:
                # Check 4-connected neighborhood
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0 or abs(dr) + abs(dc) > 1:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not touched[nr, nc]:
                                
                                for i in range(0, len(segments)):
                                    key = list(tuple(color))
                                    key.append(i)

                                    neighbor_segment = segments[i]
                                    if neighbor_segment[1][nr, nc]:
                                        neighbor_key = list(tuple(image[nr, nc]))
                                        neighbor_key.append(i)
                                        neighbor_color = tuple(neighbor_key)
                                        if neighbor_color != tuple(key):
                                            if neighbor_color not in neighbors:
                                                neighbors[neighbor_color] = neighbor_segment[2] # area
                                            touched[nr, nc] = True
                                    
    if not neighbors:
        return None, 0

    # Find the largest neighbor
    largest_neighbor = max(neighbors, key=neighbors.get)
    return segments[largest_neighbor[3]], largest_neighbor[3]

def find_small_segments(image, min_area):
    small_segments = []
    rows, cols, _ = image.shape
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    
    for color in unique_colors:
        mask = np.all(image == color, axis=-1).astype(np.uint8)  # Ensure mask is uint8
        num_labels, labeled_mask = cv2.connectedComponents(mask)
        
        # Process each connected component
        for label in range(1, num_labels):  # Skip label 0 as it's the background
            submask = (labeled_mask == label).astype(np.uint8)
            area = np.sum(submask)
            if area < min_area:
                small_segments.append((color, submask))
                return small_segments
    
    return small_segments

def get_all_segments(image):
    segments = []
    rows, cols, _ = image.shape
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    for color in unique_colors:
        mask = np.all(image == color, axis=-1).astype(np.uint8)  # Ensure mask is uint8
        num_labels, labeled_mask = cv2.connectedComponents(mask)

         # Process each connected component
        for label in range(1, num_labels):  # Skip label 0 as it's the background
            submask = (labeled_mask == label).astype(np.uint8)
            area = np.sum(submask)
            # print("segments index: " + str(len(segments)) + " has area " + str(area))
            segments.append([color, submask, area])
    
    return segments

def paint_segment(image, segment):
    rows, cols, _ = image.shape
    color = segment[0]
    submask = segment[1]
    
    # Get the bounding box of the submask to limit search area
    bbox = cv2.boundingRect(submask.astype(np.uint8))
    x, y, w, h = bbox
    submask_region = submask[y:y+h, x:x+w]
    
    for r in range(y, y+h):
        for c in range(x, x+w):
            if submask_region[r-y, c-x]:
                image[r, c] = color

def merge_segment(fromSegment, toSegment):
    toSegment[1] = fromSegment[1] | toSegment[1]
    toSegment[2] = fromSegment[2] + toSegment[2]
    toSegment[0] = fromSegment[0] if toSegment[2] < fromSegment[2] else toSegment[0] 

    return toSegment


def merge_segments(image, min_area, axis, figure):
    segments = get_all_segments(image)
    print(f"Found {len(segments)} segments")
    
    foundSmallArea = True
    while foundSmallArea:
        foundSmallArea = False
        for i in range(len(segments)-1, -1, -1):

            segment = segments[i]

            if segment[2] < min_area:
                print("Segment index " + str(i) + " has to small area " + str(segment[2]))
                neighbor_segment, index = find_largest_touching_segment(image, segment, segments)
                if neighbor_segment and index != i and index < len(segments) and index >= 0:
                    print(f"Merged segment {i} with segment index {index}")

                    segments[index] = merge_segment(segment, neighbor_segment)
                    paint_segment(image, segments[index])
                    del segments[i]
            else:
                print(f"Segment {i} is OK")
            

            if i % 50 == 0:
                axis.imshow(image)
                figure.canvas.draw()
                figure.canvas.flush_events()
                plt.pause(0.001)
                

    
    iter = 0
    for s in segments:
        paint_segment(image, s)
        # print(f"Draw segment {iter}")
        iter = iter + 1
    axis.imshow(image)
    figure.canvas.draw()
    figure.canvas.flush_events()

    return image

if __name__ == '__main__':

    image_name = 'Boat.jpg'

    # Load the image
    image = cv2.imread(image_name)

    # Define scaling factors
    target_x = 500

    scale_x = target_x / image.shape[1]
    scale_y = scale_x

    # Calculate new dimensions
    new_width = int(image.shape[1] * scale_x)
    new_height = int(image.shape[0] * scale_y)

    # Resize image
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create subplots
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Plot the original image
    axs[0].imshow(image_rgb)
    axs[0].set_title(image_name)
    axs[1].set_title(image_name + " BBSBS")
    axs[1].set_title(image_name + " SBS")
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Remove ticks from the subplot
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    img2 = image_rgb.copy()

    imageWidth = img2.shape[1] #Get image width
    imageHeight = img2.shape[0] #Get image height


    axs[1].imshow(img2)
    fig.canvas.draw()
    fig.canvas.flush_events()


    showMiddle = False
    showRight = True
    threshold1 = 1
    threshold2 = 1
    min_area = 35  # Define a minimum area for small segments


    if showMiddle:

        # Gaussian Blur
        print("Blur 1")
        img2 = cv2.GaussianBlur(img2, (7, 7), 0)
        axs[1].imshow(img2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)


        print("Blur 2")
        img2 = cv2.GaussianBlur(img2, (7, 7), 0)
        axs[1].imshow(img2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("To HSV")
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        axs[1].imshow(img2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Blur Values")
        img2 = gaussian_blur_selected_channels(img2, ksize=7, sigmaX=0, channels_to_blur=[2])
        # img2 = cv2.GaussianBlur(img2, (13, 13), 0)
        axs[1].imshow(img2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)


        print("Process")
        img2 = parallel_process_all_pixels(img2, axs[1])
        axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_HSV2RGB))
        fig.canvas.draw()
        fig.canvas.flush_events()


        print("TO RGB")
        img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2RGB)
        axs[1].imshow(img2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("Middle done!")
        plt.pause(2)

        print("Remove small spots")
        img2 = merge_segments(img2, min_area, axs[0], fig)
        axs[0].imshow(img2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("Middle segments merged!")
        plt.pause(3)

        # print("Remove small protrusions")
        # img2 = merge_protrusions_with_largest_neighbor(img2, min_area, axs[0], fig)
        # axs[0].imshow(img2)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # print("Middle protrusions merged!")
        # plt.pause(3)


        print("Edge Detection Middle")
        edges, edgesShow = apply_canny_edge_detection(img2, threshold1, threshold2)
        axs[1].imshow(edgesShow)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Edge Overlay Middle")
        img2 = overlay_edges_on_image(img2, threshold1, threshold2)
        axs[1].imshow(img2)
        fig.canvas.draw()
        fig.canvas.flush_events()

        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        filename = 'Middle.png'
        cv2.imwrite(filename, img2)

        filename = 'MiddleLines.png'
        cv2.imwrite(filename, edgesShow)
        plt.pause(1)





    ## ================= RIGHT ====================
    if showRight:
        img3 = image_rgb.copy()
        axs[1].imshow(img3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Right Gaussian Blur")
        img3 = cv2.GaussianBlur(img3, (3, 3), 0)
        axs[1].imshow(img3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Right To HSV")
        img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV)
        axs[1].imshow(img3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Process Right")
        img3 = parallel_process_all_pixels(img3, axs[1])
        axs[1].imshow(cv2.cvtColor(img3, cv2.COLOR_HSV2RGB))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Blur Values & Saturation")
        img3 = gaussian_blur_selected_channels(img3, ksize=3, sigmaX=0, channels_to_blur=[1, 2])
        axs[1].imshow(cv2.cvtColor(img3, cv2.COLOR_HSV2RGB))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Process Right")
        img3 = parallel_process_all_pixels(img3, axs[1])
        axs[1].imshow(cv2.cvtColor(img3, cv2.COLOR_HSV2RGB))
        fig.canvas.draw()
        fig.canvas.flush_events()

        print("TO RGB Right")
        img3 = cv2.cvtColor(img3, cv2.COLOR_HSV2RGB)
        axs[1].imshow(img3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(2)

        print("Remove small spots RIGHT")
        img3 = merge_segments(img3, min_area, axs[0], fig)
        axs[1].imshow(img3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("segments merged Right!")
        plt.pause(3)

        # print("Remove small protrusions RIGHT")
        # img3 = merge_protrusions_with_largest_neighbor(img3, min_area, axs[0], fig)
        # axs[0].imshow(img3)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # print("protrusions merged RIGHT!")
        # plt.pause(3)

        print("Edge Detection Right")
        edges, edgesShow = apply_canny_edge_detection(img3, threshold1, threshold2)
        axs[1].imshow(edgesShow)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        print("Edge Overlay Right")
        img3 = overlay_edges_on_image(img3, threshold1, threshold2)
        axs[1].imshow(img3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1)

        
        img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        filename = 'Right.png'
        cv2.imwrite(filename, img3)

        filename = 'RightLines.png'
        cv2.imwrite(filename, edgesShow)
        print("DONE Right")

    print("DONE!")
    plt.ioff()
    plt.show()

