import numpy as np
import cv2
import tensorflow as tf


IMAGE_SHAPE = (128, 64, 3)


class ImageEncoder(object):
    def __init__(self, checkpoint_filename, output_name='features'):
        imported = tf.saved_model.load_v2('model_data')
        self.f = imported.signatures['serving_default']
        self.output_name = output_name

    def __call__(self, data_x):
        labeling = self.f(tf.convert_to_tensor(data_x))
        return labeling[f'net/{self.output_name}:0'].numpy()


class BoxEncoder(object):
    def __init__(self, model_filename, output_name='features'):
        self.image_encoder = ImageEncoder(model_filename, output_name)

    def extract_image_patch(self, image, bbox, patch_shape):
        """Extract image patch from bounding box.

        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.

        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.

        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image

    def __call__(self, image, boxes):
        image_patches = []
        for box in boxes:
            patch = self.extract_image_patch(image, box, IMAGE_SHAPE[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(0., 255., IMAGE_SHAPE).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return self.image_encoder(image_patches)