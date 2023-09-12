import colorsys
from PIL import Image,ImageDraw
import numpy as np

#shape parameters
num_toparcs = 7
num_subarcs = 5
recursions = 4
angle = np.pi*0.95

show_all_iterations = True
show_complete_circles = True

def compute_arc(end1, end2, angle):
    '''end1 and end2 are tuples of (x,y) coordinates corresponding to the endpoints of the arc.
    angle is the angle of the arc in radians.
    end1 and end2 must be specified in clockwise order.'''
    midpoint = ((end1[0] + end2[0]) / 2, (end1[1] + end2[1]) / 2)
    chordlength = np.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2)
    halfsin = np.sin(angle / 2)
    radius = chordlength / (2 * halfsin)
    midangle = np.arctan2(end2[1] - end1[1], end2[0] - end1[0]) + np.pi / 2
    offset = np.sqrt(radius**2 - (chordlength / 2)**2)
    if angle > np.pi:
        offset *= -1

    center = (midpoint[0] + offset * np.cos(midangle), midpoint[1] + offset * np.sin(midangle))
    startangle = np.arctan2(end1[1] - center[1], end1[0] - center[0])
    endangle = np.arctan2(end2[1] - center[1], end2[0] - center[0])

    return (center, radius, startangle, endangle)

def draw_arc(draw, center, radius, startangle, endangle, color=(255, 255, 255), remainder_color = None):
    if remainder_color is not None:
        draw.arc((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), endangle*180.0/np.pi, startangle*180.0/np.pi, remainder_color)
    draw.arc((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), startangle*180.0/np.pi, endangle*180.0/np.pi, color)


def woven_circles(center, radius, num_toparcs, angle, num_subarcs, depth):
    def fractal_arc(end1, end2, angle, num_arcs, depth):
        arc = compute_arc(end1, end2, angle)
        if depth < 0:
            return [], []
        if depth == 0:
            return [arc], []
        else:
            dtheta = angle / float(num_arcs)
            final_arcs = []
            iterative_arcs = [arc]
            for i in range(num_arcs):
                theta = i * dtheta + arc[2]
                newend1 = (arc[0][0] + arc[1] * np.cos(theta), arc[0][1] + arc[1] * np.sin(theta))
                newend2 = (arc[0][0] + arc[1] * np.cos(theta + dtheta), arc[0][1] + arc[1] * np.sin(theta + dtheta))
                finarcs, itarcs = fractal_arc(newend1, newend2, angle, num_arcs, depth - 1)
                final_arcs += finarcs
                iterative_arcs += itarcs
            return final_arcs, iterative_arcs

    dtheta = 2 * np.pi / num_toparcs
    final_arcs = []
    iterative_arcs = [(center, radius, 0, np.pi * 1.99)]
    for i in range(num_toparcs):
        theta = i * dtheta
        end1 = (center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta))
        end2 = (center[0] + radius * np.cos(theta + dtheta), center[1] + radius * np.sin(theta + dtheta))
        finarcs, itarcs = fractal_arc(end1, end2, angle, num_subarcs, depth - 1)
        final_arcs += finarcs
        iterative_arcs += itarcs
    return final_arcs, iterative_arcs

def draw_arcs(draw, arcs, color, extra_color=None):
    for arc in arcs:
        draw_arc(draw, arc[0], arc[1], arc[2], arc[3], color, extra_color)

img = Image.new('RGB', (1000, 1000))
draw = ImageDraw.Draw(img)

#draw_arc(draw, (500, 500), 100, 0, np.pi*2.0/3.0, (255, 0, 0))
final_arcs, iterative_arcs = woven_circles((500, 500), 300, num_toparcs, angle, num_subarcs, recursions)

extra_color = (64, 64, 64) if show_complete_circles else None
if show_all_iterations:
    draw_arcs(draw, iterative_arcs, (128, 0, 255), extra_color)
draw_arcs(draw, final_arcs, (255, 200, 0), extra_color)

img.show()