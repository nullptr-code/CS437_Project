import matplotlib.pyplot as plt


def showImage(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def showImageRow(images, title=None):
    plt.figure(figsize=(17, 15))
    if title is not None:
        plt.title(title)
    for i, image in enumerate(images, 1):
        plt.subplot(1, 5, i)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def threshold(image, thres=0.5):
    image[image > thres] = 1
    image[image <= thres] = 0
    return image