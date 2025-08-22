import matplotlib.pyplot as plt


def id_to_color(idx: int):
    blue = idx * 5 % 256
    green = idx * 12 % 256
    red = idx * 23 % 256
    return (red, green, blue)


def visualize_images(images):
    n = len(images)
    fig, axs = plt.subplots(n, 1, figsize=(10, 5))
    for index, image in enumerate(images):
        axs[index].imshow(image)
        axs[index].axis("off")

    plt.tight_layout()
    plt.show()


def division_by_zero(value, epsilon=0.01):
    if value < epsilon:
        value = epsilon
    return value
