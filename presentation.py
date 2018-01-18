import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import imageio
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc


def create_heat_map_image_from_label(label, mappable=None):
    if not mappable:
        mappable = matplotlib.cm.ScalarMappable(cmap='inferno')
    heat_map_image = mappable.to_rgba(label, bytes=True)
    return heat_map_image


def generate_heat_map_video_from_labels(labels, output_directory, filter_roi=True):
    writer = imageio.get_writer(os.path.join(output_directory, 'heat_map.mp4'), fps=50)
    rois = np.load(os.path.join(output_directory, 'rois.npy'), mmap_mode='r')
    mappable = matplotlib.cm.ScalarMappable(cmap='inferno')
    mappable.set_clim(vmin=min(labels.min(), labels.min()),
                      vmax=max(labels.max(), labels.max()))
    for index, label in enumerate(labels):
        heat_map = create_heat_map_image_from_label(label, mappable)
        if filter_roi:
            roi = np.bitwise_not(rois[index])
            heat_map[roi, :] = [0, 80, 0, 1]
        writer.append_data(heat_map)
    writer.close()


def generate_video_from_images(images, output_directory):
    writer = imageio.get_writer(os.path.join(output_directory, 'video.mp4'), fps=50)
    for image in images:
        writer.append_data(image)
    writer.close()


def generate_heat_map_overlay_video_from_data_directory(data_directory, color_perspective_scaled=True):
    print('Creating image video...')
    images = np.load(os.path.join(data_directory, 'images.npy'), mmap_mode='r')
    generate_video_from_images(images, data_directory)
    print('Creating heatmap video...')
    labels = np.load(os.path.join(data_directory, 'predicted_labels.npy'), mmap_mode='r')
    if color_perspective_scaled:
        perspectives = np.load(os.path.join(data_directory, 'perspectives.npy'), mmap_mode='r')
        perspective_scalar = 0.1
        labels = labels + (labels * perspectives * perspective_scalar)
    generate_heat_map_video_from_labels(labels, data_directory)
    print('Combining to overlay video...')
    generate_overlay_video(data_directory)


def generate_overlay_video(data_directory):
    video_reader = imageio.get_reader(os.path.join(data_directory, 'video.mp4'))
    heat_map_reader = imageio.get_reader(os.path.join(data_directory, 'heat_map.mp4'))
    overlay_writer = imageio.get_writer(os.path.join(data_directory, 'overlay.mp4'), fps=50)
    heat_map_iter = iter(heat_map_reader)
    for image in video_reader:
        heat_map = next(heat_map_iter)
        stack = np.stack([image, heat_map])
        overlay = np.average(stack, axis=0, weights=[0.5, 0.5]).clip(0, 255).astype(np.uint8)
        overlay_writer.append_data(overlay)
    overlay_writer.close()


def generate_plot_video(data_directory):
    sns.set_style('darkgrid')
    labels = np.load(os.path.join(data_directory, 'predicted_labels.npy'), mmap_mode='r')
    rois = np.load(os.path.join(data_directory, 'rois.npy'), mmap_mode='r')
    length = labels.shape[0]
    roi_labels = labels * rois
    counts = roi_labels.sum(axis=(1, 2))
    plot_writer = imageio.get_writer(os.path.join(data_directory, 'plot.mp4'), fps=50)
    for index in range(length):
        figure, axes = plt.subplots(dpi=300)
        axes.set_xlim(0, length)
        axes.set_ylim(0, counts.max())
        axes.plot(np.arange(index), counts[:index], color=sns.color_palette()[0])
        axes.set_ylabel('Person Count')
        axes.set_xlabel('Frame')
        figure.canvas.draw()
        image = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        plot_writer.append_data(image)
        plt.close(figure)
    plot_writer.close()


def position_videos_side_by_side(data_directory):
    overlay_reader = imageio.get_reader(os.path.join(data_directory, 'overlay.mp4'))
    plot_reader = imageio.get_reader(os.path.join(data_directory, 'plot.mp4'))
    complete_writer = imageio.get_writer(os.path.join(data_directory, 'complete.mp4'), fps=50)
    plot_reader_iter = iter(plot_reader)
    for overlay in overlay_reader:
        plot = next(plot_reader_iter)
        overlay = scipy.misc.imresize(overlay, (plot.shape[0], int(overlay.shape[1] * (plot.shape[0]/overlay.shape[0]))))
        combined = np.concatenate([overlay, plot], axis=1)
        complete_writer.append_data(combined)
    complete_writer.close()


if __name__ == '__main__':
    data_directory_ = '../data/104207 Normal Speed Demo'
    #data_directory_ = 'data/mini_world_expo_datasets/validation'
    print('# Creating Heatmap Overlay Video')
    generate_heat_map_overlay_video_from_data_directory(data_directory_)
    print('# Creating Plot Video')
    generate_plot_video(data_directory_)
    print('# Combining Videos')
    position_videos_side_by_side(data_directory_)
