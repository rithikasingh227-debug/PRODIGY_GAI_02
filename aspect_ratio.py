import keras_cv
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusion(768, 512)

images = model.text_to_image("A sunset over mountains")

plt.imshow(images[0])
plt.axis("off")
plt.savefig("aspect_output.png")
