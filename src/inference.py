import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import argparse
import os
import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    print('Tensorflow 2.x.x required')
    sys.exit(1)

import unet_model
import numpy as np
import skimage.io


def zscore_normalize(image_data):
    image_data = image_data.astype(np.float32)

    if len(image_data.shape) == 3:
        # input is CHW
        for c in range(image_data.shape[0]):
            std = np.std(image_data[c, :, :])
            mv = np.mean(image_data[c, :, :])
            if std <= 1.0:
                # normalize (but dont divide by zero)
                image_data[c, :, :] = (image_data[c, :, :] - mv)
            else:
                # z-score normalize
                image_data[c, :, :] = (image_data[c, :, :] - mv) / std
    elif len(image_data.shape) == 2:
        # input is HW
        std = np.std(image_data)
        mv = np.mean(image_data)
        if std <= 1.0:
            # normalize (but dont divide by zero)
            image_data = (image_data - mv)
        else:
            # z-score normalize
            image_data = (image_data - mv) / std
    else:
        raise IOError("Input to Z-Score normalization needs to be either a 2D or 3D image [HW, or CHW]")

    return image_data


def _inference_tiling(img_filepath, model, tile_size):

    print('Loading image: {}'.format(img_filepath))
    img = skimage.io.imread(img_filepath)
    img = img.astype(np.float32)

    # normalize with whole image stats
    img = zscore_normalize(img)
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros(img.shape, dtype=np.int32)
    print('  img.shape={}'.format(img.shape))

    radius = unet_model.UNet.SIZE_FACTOR
    zone_of_responsibility_size = tile_size - 2 * radius
    for i in range(0, height, zone_of_responsibility_size):
        for j in range(0, width, zone_of_responsibility_size):

            x_st_z = j
            y_st_z = i
            x_end_z = x_st_z + zone_of_responsibility_size
            y_end_z = y_st_z + zone_of_responsibility_size

            # pad zone of responsibility by radius
            x_st = x_st_z - radius
            y_st = y_st_z - radius
            x_end = x_end_z + radius
            y_end = y_end_z + radius

            pre_pad_x = 0
            if x_st < 0:
                pre_pad_x = -x_st
                x_st = 0
            pre_pad_y = 0
            if y_st < 0:
                pre_pad_y = -y_st
                y_st = 0
            post_pad_x = 0
            if x_end > width:
                post_pad_x = x_end - width
                x_end = width
            post_pad_y = 0
            if y_end > height:
                post_pad_y = y_end - height
                y_end = height

            # crop out the tile
            tile = img[y_st:y_end, x_st:x_end]

            if pre_pad_x > 0 or pre_pad_y > 0 or post_pad_x > 0 or post_pad_y > 0:
                # ensure its correct size (if tile exists at the edge of the image
                tile = np.pad(tile, pad_width=((pre_pad_y, post_pad_y), (pre_pad_x, post_pad_x)), mode='reflect')

            if len(tile.shape) == 2:
                # add a channel dimension
                tile = tile.reshape((tile.shape[0], tile.shape[1], 1))

            # convert HWC to CHW
            batch_data = tile.transpose((2, 0, 1))
            # convert CHW to NCHW
            batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))

            softmax = model(batch_data) # model output defined in unet_model is softmax
            pred = np.squeeze(np.argmax(softmax, axis=-1).astype(np.int32))

            pre_pad_x = max(pre_pad_x, radius)
            if pre_pad_x > 0:
                pred = pred[:, pre_pad_x:]
            pre_pad_y = max(pre_pad_y, radius)
            if pre_pad_y > 0:
                pred = pred[pre_pad_y:, :]
            post_pad_x = max(post_pad_x, radius)
            if post_pad_x > 0:
                pred = pred[:, :-post_pad_x]
            post_pad_y = max(post_pad_y, radius)
            if post_pad_y > 0:
                pred = pred[:-post_pad_y, :]

            mask[y_st_z:y_end_z, x_st_z:x_end_z] = pred

    return mask


def _inference(img_filepath, model):

    print('Loading image: {}'.format(img_filepath))
    img = skimage.io.imread(img_filepath)
    img = img.astype(np.float32)

    # normalize with whole image stats
    img = zscore_normalize(img)

    print('  img.shape={}'.format(img.shape))
    pad_x = 0
    pad_y = 0

    if img.shape[0] % unet_model.UNet.SIZE_FACTOR != 0:
        pad_y = (unet_model.UNet.SIZE_FACTOR - img.shape[0] % unet_model.UNet.SIZE_FACTOR)
        print('image height needs to be a multiple of {}, padding with reflect'.format(unet_model.UNet.SIZE_FACTOR))
    if img.shape[1] % unet_model.UNet.SIZE_FACTOR != 0:
        pad_x = (unet_model.UNet.SIZE_FACTOR - img.shape[1] % unet_model.UNet.SIZE_FACTOR)
        print('image width needs to be a multiple of {}, padding with reflect'.format(unet_model.UNet.SIZE_FACTOR))
    if pad_x > 0 or pad_y > 0:
        img = np.pad(img, pad_width=((0, pad_y), (0, pad_x)), mode='reflect')

    if len(img.shape) == 2:
        # add a channel dimension
        img = img.reshape((img.shape[0], img.shape[1], 1))

    # convert HWC to CHW
    batch_data = img.transpose((2, 0, 1))
    # convert CHW to NCHW
    batch_data = batch_data.reshape((1, batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))

    softmax = model(batch_data) # model output defined in unet_model is softmax
    pred = np.squeeze(np.argmax(softmax, axis=-1).astype(np.int32))

    if pad_x > 0:
        pred = pred[:, 0:-pad_x]
    if pad_y > 0:
        pred = pred[0:-pad_y, :]

    return pred


def inference(saved_model_filepath, image_folder, output_folder, image_format, tile_size):
    # create output filepath
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    img_filepath_list = [os.path.join(image_folder, fn) for fn in os.listdir(image_folder) if fn.endswith('.{}'.format(image_format))]

    model = tf.saved_model.load(saved_model_filepath)

    print('Starting inference of file list')
    for i in range(len(img_filepath_list)):
        img_filepath = img_filepath_list[i]
        _, slide_name = os.path.split(img_filepath)
        print('{}/{} : {}'.format(i, len(img_filepath_list), slide_name))

        if tile_size > 0:
            segmented_mask = _inference_tiling(img_filepath, model, tile_size)
        else:
            segmented_mask = _inference(img_filepath, model)

        if 0 <= np.max(segmented_mask) <= 255:
            segmented_mask = segmented_mask.astype(np.uint8)
        if 255 < np.max(segmented_mask) < 65536:
            segmented_mask = segmented_mask.astype(np.uint16)
        if np.max(segmented_mask) > 65536:
            segmented_mask = segmented_mask.astype(np.int32)
        skimage.io.imsave(segmented_mask, os.path.join(output_folder, slide_name))


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='inference', description='Script which inferences a folder of images using a unet model')

# TODO set the saved model inptu up as a collection
    parser.add_argument('--savedModel', dest='saved_model_filepath', type=str,
                        help='SavedModel filepath to the  model to use', required=True)
    parser.add_argument('--imageDir', dest='image_dir', type=str, help='filepath to the directory containing the images', required=True)
    parser.add_argument('--outputDir', dest='output_dir', type=str, help='Folder where outputs will be saved (Required)', required=True)
    parser.add_argument('--useTiling', dest='use_tiling', type=str, help='whether to use tiling when training [YES, NO]', default="NO")
    parser.add_argument('--tileSize', dest='tile_size', type=int, default=256)


    print('Arguments:')
    args = parser.parse_args()

    saved_model_filepath = args.saved_model_filepath
    use_tiling = args.use_tiling
    use_tiling = use_tiling.upper() == "YES"
    output_dir = args.output_dir
    tile_size = args.tile_size
    image_dir = args.image_dir

    print('use_tiling = {}'.format(use_tiling))
    print('tile_size = {}'.format(tile_size))
    print('image_dir = {}'.format(image_dir))
    print('output_dir = {}'.format(output_dir))

    image_format = 'tif'
    # zero out tile size with its turned off
    if not use_tiling:
        # tile_size <= 0 disables tiling
        tile_size = 0
    else:
        assert tile_size % unet_model.UNet.SIZE_FACTOR == 0, 'UNet requires tiles with shapes that are multiples of 16'

    inference(saved_model_filepath, image_dir, output_dir, image_format, tile_size)


if __name__ == "__main__":
    main()