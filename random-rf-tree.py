import colorsys
from PIL import Image,ImageDraw
import numpy as np

w = 128
h = 128
img = Image.new("RGB", (w,h))
draw = ImageDraw.Draw(img)

def gaussian_tree_2d(mean, var, bounds, child_num, depth, scale_factor):
    def sample(mean, var):
        return np.random.multivariate_normal(mean, var)
        # TO DO: evaluate if it's faster to choose sample polarly. Choose angle uniformly, then magnitude from a normal. Since we probably want to compute the positional embedding in that format anyway.

    loc = sample(mean, var)

    while loc[0] < bounds[0] or loc[1] < bounds[1] or loc[0] > bounds[2] or loc[1] > bounds[3]:
        loc = sample(mean, var)

    # convert to tuple
    loc = (loc[0], loc[1])

    if depth > 1:
        points = []
        for i in range(child_num):
            new_points = gaussian_tree_2d(loc, var*scale_factor, bounds, child_num, depth - 1, scale_factor)
            points.append(new_points)
        return points
    else:
        return loc

def render_points_grouped(img, points, groups):
    def rainbow(x, xmax):
        t = colorsys.hsv_to_rgb(x / float(xmax), 1.0, 1.0)
        return tuple(int(c*255) for c in t)

    for i,point in enumerate(points):
        draw.point(point, rainbow((i*groups)//len(points), groups))


child_num = 16
depth = 4
scale_factor = 0.25

s = min(w, h)
points = gaussian_tree_2d((w/2, h/2), np.identity(2)*s*4.0, (0, 0, w-1, h-1), child_num, depth, scale_factor)
flatpoints = np.reshape(points, (-1, 2)).tolist()

render_points_grouped(img, flatpoints, 16)#child_num**(depth - 1))
img.show()