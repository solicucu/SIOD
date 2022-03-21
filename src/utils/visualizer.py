# Copyright (c) Facebook, Inc. and its affiliates.
import colorsys
import json
import argparse
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from colormap import random_color

_SMALL_OBJECT_AREA_THRESH = 1000
class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

class Visualizer:
    def __init__(self, img_path, scale=1.0, prefix="new"):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.img = img_rgb 

        self.output = VisImage(self.img, scale=scale)
        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        names = img_path.split("/")
        names[-1] = "{}_{}".format(prefix,names[-1])
        self.save_path = "/".join(names)
    
    def save(self):
        self.output.save(self.save_path)

    """
    Args:
        boxes(ndarray): [n, 4], [x1,y1,x2,y2]
        labels(list): [n]
        colors(list(tuple)):[n,3]
    """
    def draw_boxes_with_labels(self, boxes, labels, colors=None):
        num = len(boxes)
        for i in range(num):
            color = colors[i]
            self.draw_box(boxes[i], edge_color=color)
            x0, y0, x1, y1 = boxes[i]
            text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
            horiz_align = "left"
            # for small objects, draw text at the side to avoid occlusion
            instance_area = (y1 - y0) * (x1 - x0)
            if (
                instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                or y1 - y0 < 40 * self.output.scale
            ):
                if y1 >= self.output.height - 5:
                    text_pos = (x1, y0)
                else:
                    text_pos = (x0, y1)
            
            height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * self._default_font_size
            )
            self.draw_text(
                labels[i],
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
            )

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            horizontal_alignment="center",
            rotation=0
        ):
            """
            Args:
                text (str): class label
                position (tuple): a tuple of the x and y coordinates to place text on image.
                font_size (int, optional): font of the text. If not provided, a font size
                    proportional to the image width is calculated and used.
                color: color of the text. Refer to `matplotlib.colors` for full list
                    of formats that are accepted.
                horizontal_alignment (str): see `matplotlib.text.Text`
                rotation: rotation angle in degrees CCW

            Returns:
                output (VisImage): image object with text drawn.
            """
            if not font_size:
                font_size = self._default_font_size

            # since the text background is dark, we don't want the text to be dark
            color = np.maximum(list(mplc.to_rgb(color)), 0.2)
            color[np.argmax(color)] = max(0.8, np.max(color))

            x, y = position
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
            return self.output

    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def draw_box(self, box_coord, alpha=1., edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output  

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
      82, 84, 85, 86, 87, 88, 89, 90]
continuous_ids = {k:v for v,k in enumerate(valid_ids)}

def load_detect_results_and_visualize(path, result_file, thresh=0.3, prefix=""):
    with open(result_file, 'r') as f:
        data = json.load(f)
    """
    data:{"image_name":{class_id:[[x1,y1,x2,y2,score],..],...}
    """
    # name is image_name, value is a dict
    for name, value in data.items():
        boxes = []
        labels = []
        colors = []
        for cid, item in value.items():
            # [k,5]
            box_score = np.asarray(item)
            score = box_score[:,4]
            mask = score > thresh
            selects = box_score[mask]
            if len(selects):
                boxes.append(selects[:,:4])
                cid = int(cid)-1
                label = ["{} {:.2f}".format(class_names[cid], score) for score in selects[:,4]]
                color = [random_color(rgb=True, maximum=1)] * len(label)
                # color = [COLORS[cid]]*len(label)
                labels.extend(label)
                colors.extend(color)

        image_path = path + name
        vis = Visualizer(image_path, prefix=prefix)
        # [n, 4]
        if len(boxes) > 0:
            boxes = np.vstack(boxes).astype(np.int32)
            # labels: [n]
            # draw the boxes
            vis.draw_boxes_with_labels(boxes,labels,colors)
        vis.save()

def load_annotation_and_visualize(path, anno_file, prefix="", key="new"):
    with open(anno_file, 'r') as f:
        data = json.load(f)
    data = data[key]
    for name, annos in data.items():
        boxes = []
        labels = []
        colors = []
        color_dict = {}
        for ann in annos:
            box = ann["bbox"]
            box[2] += box[0] # x2
            box[3] += box[1] # y2
            cid = continuous_ids[ann["category_id"]]
            if cid not in color_dict:
                color = random_color(rgb=True, maximum=1)
                color_dict[cid] = color
            colors.append(color_dict[cid])
            labels.append(class_names[cid])
            boxes.append(box)
        image_path = path + name
        # draw the boxes
        vis = Visualizer(image_path, prefix=prefix)
        vis.draw_boxes_with_labels(boxes, labels, colors)
        vis.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_path", type=str, default='',help='path to the images')
    parser.add_argument("--result_file", type=str, default='', help="detection result file from centernet demo")
    parser.add_argument("--prefix", type=str, default='result', help="add prefix to the target image")
    parser.add_argument("--thresh", type=float, default=0.3, help="score threshold for visualization")

    args = parser.parse_args()
    load_detect_results_and_visualize(args.imgs_path, args.result_file, thresh=args.thresh, prefix=args.prefix)
    # load_annotation_and_visualize(args.imgs_path, args.result_file, prefix=args.prefix, key="old")

if __name__ == "__main__":
    main()
