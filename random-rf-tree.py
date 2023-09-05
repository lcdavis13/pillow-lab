import colorsys
from PIL import Image,ImageDraw
import numpy as np
from pathlib import Path

w = 128
h = 128
img = Image.new("RGB", (w,h))
draw = ImageDraw.Draw(img)


def gaussian_tree_2d(mean, cov, bounds, child_num, depth, scale_factor):
    def sample(mean, var):
        return np.random.multivariate_normal(mean, var)
        # TO DO: evaluate if it's faster to choose sample polarly. Choose angle uniformly, then magnitude from a normal. Since we probably want to compute the positional embedding in that format anyway.

    loc = sample(mean, cov)

    while loc[0] < bounds[0] or loc[1] < bounds[1] or loc[0] > bounds[2] or loc[1] > bounds[3]:
        loc = sample(mean, cov)

    # convert to tuple
    loc = (loc[0], loc[1])

    if depth > 1:
        points = []
        for i in range(child_num):
            new_points = gaussian_tree_2d(loc, cov * scale_factor, bounds, child_num, depth - 1, scale_factor)
            points.append(new_points)
        return points
    else:
        return loc


def render_points_grouped(draw, points, groups):
    def rainbow(x, xmax):
        t = colorsys.hsv_to_rgb(x / float(xmax), 1.0, 1.0)
        return tuple(int(c*255) for c in t)

    for i,point in enumerate(points):
        draw.point(point, rainbow((i*groups)//len(points), groups))

def render_tree(resolution, tree):
    w = resolution[0]
    h = resolution[1]

    image_tree = {}

    levels = len(np.shape(tree))
    if levels > 1:
        img = Image.new("RGB", (w,h))
        draw = ImageDraw.Draw(img)

        points = np.reshape(tree, (-1, 2)).tolist()
        render_points_grouped(draw, points, np.shape(tree)[0])

        image_tree["img"] = img

    if levels > 2:
        subtrees = []
        for i,subtree in enumerate(tree):
            subtree = render_tree(resolution, subtree)
            subtrees.append(subtree)
        image_tree["children"] = subtrees

    return image_tree

def export_image_tree(tree, rootpath, rootname="tree"):
    def export_tree(tree, fpath, rootname, id=0):
        newrootname = rootname + "_" + str(id)
        if "img" in tree:
            fname = fpath + "/" + newrootname + ".png"
            tree["img"].save(fname)
        if "children" in tree:
            newpath = fpath + "/" + str(id)
            Path(newpath).mkdir(parents=True, exist_ok=True)
            for i,child in enumerate(tree["children"]):
                export_tree(child, newpath, newrootname, i)
    Path(rootpath).mkdir(parents=True, exist_ok=True)
    export_tree(tree, rootpath, rootname)



child_num = 16
depth = 4
scale_factor = 0.25

s = min(w, h)
tree = gaussian_tree_2d(mean=(w/2, h/2),
                        cov=np.identity(2) * s * 4.0,
                        bounds=(0, 0, w - 1, h - 1),
                        child_num=child_num, depth=depth, scale_factor=scale_factor)
image_tree = render_tree((w,h), tree)
export_image_tree(image_tree, "./tree")

flatpoints = np.reshape(tree, (-1, 2)).tolist()

render_points_grouped(draw, flatpoints, 16)#child_num**(depth - 1))
img.show()