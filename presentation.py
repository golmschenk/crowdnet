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
    counts = savitzky_golay(counts, 7, 3)
    plot_writer = imageio.get_writer(os.path.join(data_directory, 'plot.mp4'), fps=50)
    for index in range(length):
        print('\rPlotting {}...'.format(index), end='')
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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


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
    data_directory_ = '../data/200608 Time Lapse Demo'
    #data_directory_ = 'data/mini_world_expo_datasets/validation'
    print('# Creating Heatmap Overlay Video')
    generate_heat_map_overlay_video_from_data_directory(data_directory_)
    print('# Creating Plot Video')
    generate_plot_video(data_directory_)
    print('# Combining Videos')
    position_videos_side_by_side(data_directory_)
