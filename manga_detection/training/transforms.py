class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target
