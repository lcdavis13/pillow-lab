import colorsys
from PIL import Image,ImageDraw
import numpy as np
from pathlib import Path


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


def draw_points_grouped(draw, points, groups, color_scheme):
    for i,point in enumerate(points):
        c = ((i*groups)//len(points)) / groups
        draw.point(point, color_scheme(c))


def draw_points(draw, points, color):
    for point in points:
        draw.point(point, color)


def draw_tree_node_rainbow(draw, tree):
    def rainbow(hue):
        t = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return tuple(int(c*255) for c in t)
    points = np.reshape(tree, (-1, 2)).tolist()
    draw_points_grouped(draw, points, np.shape(tree)[0], rainbow)


def draw_tree_node(draw, tree, color):
    points = np.reshape(tree, (-1, 2)).tolist()
    draw_points(draw, points, color)


def render_tree(resolution, tree, bg=None):
    image_tree = {}
    w = resolution[0]
    h = resolution[1]

    if bg is None:
        bg = Image.new("RGB", (w, h))

    levels = len(np.shape(tree))
    if levels > 1:
        img = bg.copy()
        draw = ImageDraw.Draw(img)
        draw_tree_node_rainbow(draw, tree)
        image_tree["img"] = img

    if levels > 2:
        # render new background
        img = np.asarray(bg, dtype="int32")
        img = img/2.0
        img = Image.fromarray(np.asarray(img, dtype="uint8"), "RGB")
        draw = ImageDraw.Draw(img)
        draw_tree_node(draw, tree, (127, 127, 127))

        # render subtrees
        subtrees = []
        for i,subtree in enumerate(tree):
            subtree = render_tree(resolution, subtree, img)
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


def animate_image_tree(tree, layer_times, fpath):
    def animate_tree_list(tree, layer_times):
        img_seq = []
        dur_seq = []
        for node in tree:
            if "img" in node:
                img_seq.append(node["img"])
                dur_seq.append(layer_times[0])
            if "children" in node:
                img_subseq, dur_subseq = animate_tree_list(node["children"], layer_times[1:])
                img_seq += img_subseq
                dur_seq += dur_subseq
        return img_seq, dur_seq

    img_seq, dur_seq = animate_tree_list([tree], layer_times)
    img_seq[0].save(fpath, save_all=True, optimize=True, append_images=img_seq[1:], duration=dur_seq, loop=0)


if __name__ == "__main__":
    w = 128
    h = 128
    child_num = 16
    depth = 4
    scale_factor = 0.25

    s = min(w, h)
    tree = gaussian_tree_2d(mean=(w/2, h/2),
                            cov=np.identity(2) * s * 4.0,
                            bounds=(0, 0, w - 1, h - 1),
                            child_num=child_num, depth=depth, scale_factor=scale_factor)
    image_tree = render_tree((w,h), tree)
    export_image_tree(image_tree, "./random-rf-tree")
    animate_image_tree(image_tree, [500, 250, 125], './random-rf-tree/tree.gif')

    image_tree["img"].show()

