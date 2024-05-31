import cv2
import numpy as np
from scipy.spatial import Delaunay
from datetime import datetime
import os
import configparser
import networkx as nx
import colorsys
from tqdm import tqdm
import glob

def average_color(image, triangle):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [triangle], -1, (255), -1, cv2.LINE_AA)
    mean_color = cv2.mean(image, mask=mask.astype(np.uint8))[:3]
    return [min(max(int(c), 0), 255) for c in mean_color]
def save_intermediate_image(
        image, 
        output_directory, 
        base_name, 
        step, 
        intermediate_gen):
    if intermediate_gen:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(
            output_directory, 
            base_name, 
            "intermediate", 
            f"{base_name}_{step}_{timestamp}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
def draw_triangle(image, triangle, color):
    cv2.drawContours(image, [triangle], -1, color, -1, cv2.LINE_AA)

def adjust_color(color, threshold, lightness):
    if color == [0, 0, 0]:  # the color is black
        return color
    hls_color = colorsys.rgb_to_hls(*[c/255 for c in color])  # normalize to 0-1 for colorsys
    l = hls_color[1]
    if l < 0.5:  # the color is dark
        l = min(1, l + lightness)
    else:  # the color is light
        l = max(0, l - lightness)
    rgb_color = colorsys.hls_to_rgb(hls_color[0], l, hls_color[2])
    return [int(c * 255) for c in rgb_color]  # denormalize from 0-1 to 0-255 for OpenCV

def divide_large_triangles(triangles, threshold, center):
    def check_triangles_format(triangles):
        for i, triangle in enumerate(triangles):
            if not isinstance(triangle, np.ndarray) or triangle.shape != (3, 2):
                print(f"Triangle {i} is not in the correct format: {triangle}")
                return False
        return True

    if not check_triangles_format(triangles):
        return []

    processed_triangles = []
    with tqdm(triangles, desc="Dividing large triangles", position=1) as t:
        for triangle in t:
            # Calculate the area of the triangle
            area = cv2.contourArea(triangle.astype(np.int32))

            if area > threshold:
                # Find the vertex closest to the center of the image
                distances = np.linalg.norm(triangle - center, axis=1)
                closest_vertex = triangle[np.argmin(distances)]

                # Find the midpoint of the opposite side
                opposite_side = triangle[triangle != closest_vertex].reshape(-1, 2)
                midpoint = np.mean(opposite_side, axis=0)

                # Divide the triangle into two smaller triangles
                processed_triangles.append(np.array([closest_vertex, opposite_side[0], midpoint]))
                processed_triangles.append(np.array([closest_vertex, opposite_side[1], midpoint]))
            else:
                processed_triangles.append(triangle)

    return processed_triangles

def image_to_triangles(
        image_path, 
        output_directory, 
        canny_threshold1, 
        canny_threshold2, 
        blur_size, 
        threshold, 
        triangle_size_threshold, 
        lightness, 
        intermediate_gen):
    print("Reading image...")
    image = cv2.imread(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    # Save the original image
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    original_output_path = os.path.join(output_directory, name, f"{name}_OG_{timestamp}{ext}")
    os.makedirs(os.path.dirname(original_output_path), exist_ok=True)
    cv2.imwrite(original_output_path, image)

    print("Adding border to image...")
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    save_intermediate_image(image, output_directory, name, 'bordered', intermediate_gen)
    
    print("Blurring image...")
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    save_intermediate_image(blurred, output_directory, name, 'blurred', intermediate_gen)
    
    print("Converting image to grayscale...")
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    save_intermediate_image(gray, output_directory, name, 'gray', intermediate_gen)
    
    print("Detecting edges...")
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
    save_intermediate_image(edges, output_directory, name, 'edges', intermediate_gen)
    
    print("Finding contours...")
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print("Processing points...")
    contour_points = np.vstack(contours).squeeze()
    height, width = image.shape[:2]
    
    # Add a contour around the border of the image
    border_points = np.array([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]])
    points = np.vstack([contour_points, border_points])

    print("Processing triangles...")
    # Generate Delaunay triangulation
    delaunay = Delaunay(points)
    
    # Convert simplices to list of triangles
    triangles = [points[simplex] for simplex in delaunay.simplices]

    # Calculate the center of the image
    center = np.array([width / 2, height / 2])

    # Calculate the triangle size threshold as a percentage of the total size of the image
    total_size = width * height
    triangle_size_threshold = total_size * (triangle_size_threshold / 100)

    print("Dividing large triangles...")
    triangles = divide_large_triangles(triangles, triangle_size_threshold, center)

    print("Initializing output image...")
    output = np.zeros(image.shape, dtype=np.uint8)

    print("Initializing graph...")
    G = nx.Graph()
    point_to_triangles = {}

    print("Processing triangle coords and mapping neighbors...")
    with tqdm(triangles, desc="Processing triangle coords", position=2) as t:
        for triangle in t:
            triangle_coords = triangle.astype(int)
            # Check if triangle_coords is not empty and is in the correct format
            if triangle_coords.size > 0 and triangle_coords.shape[1] == 2:
                color = average_color(image, triangle_coords)
                triangle_tuple = tuple(map(tuple, triangle))  # Convert numpy array to tuple of tuples
                G.add_node(triangle_tuple, coords=triangle_coords, color=color)

                # Map neighbors
                for point in triangle:
                    point_tuple = tuple(point)  # Convert numpy array to tuple
                    if point_tuple in point_to_triangles:
                        for neighbor in point_to_triangles[point_tuple]:
                            G.add_edge(triangle_tuple, neighbor)
                        point_to_triangles[point_tuple].append(triangle_tuple)
                    else:
                        point_to_triangles[point_tuple] = [triangle_tuple]

    #Adjusting colors
    with tqdm(G.nodes, desc="Adjusting colors", position=3) as t:
        for node in t:
            color = G.nodes[node]['color']
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                neighbor_color = G.nodes[neighbor]['color']
                if abs(np.array(color) - np.array(neighbor_color)).max() / 255 <= threshold:
                    G.nodes[node]['color'] = adjust_color(color, threshold, lightness)

    #Drawing triangles
    with tqdm(G.nodes, desc="Drawing triangles", position=4) as t:
        for node in t:
            triangle_coords = G.nodes[node]['coords']
            color = G.nodes[node]['color']
            draw_triangle(output, triangle_coords, color)

    print("Saving output image...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = os.path.join(output_directory, name, f"{name}_{timestamp}{ext}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    environment = config.get('DEFAULT', 'environment')
    input_directory = config.get(environment, 'input_directory')
    output_directory = config.get(environment, 'output_directory')
    canny_threshold1 = config.getint(environment, 'canny_threshold1')
    canny_threshold2 = config.getint(environment, 'canny_threshold2')
    blur_size = config.getint(environment, 'blur_size')
    color_difference_threshold = config.getfloat(environment, 'color_difference_threshold')
    lightness_adjustment = config.getfloat(environment, 'lightness_adjustment')
    intermediate_gen = config.getboolean(environment, 'intermediate_gen')
    triangle_size_threshold = config.getfloat(environment, 'triangle_size_threshold')

    image_files = glob.glob(os.path.join(input_directory, '*'))
    with tqdm(total=len(image_files), desc="Processing all images", position=0) as pbar:
        for image_file in image_files:
            print(f'Processing {image_file}:')
            image_to_triangles(
                image_file, 
                output_directory, 
                canny_threshold1, 
                canny_threshold2, 
                blur_size, 
                color_difference_threshold, 
                triangle_size_threshold, 
                lightness_adjustment, 
                intermediate_gen)
            pbar.update()