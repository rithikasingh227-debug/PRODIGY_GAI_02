import keras_cv
import matplotlib.pyplot as plt

print("Loading model...")

model = keras_cv.models.StableDiffusion(
    img_width=512,
    img_height=512
)

prompt = "A cute robot reading a book"

print("Generating image...")

images = model.text_to_image(prompt)

plt.imshow(images[0])
plt.axis("off")

plt.savefig("sd_output.png")

print("Image saved!")
