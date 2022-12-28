from PIL import Image

from imagetools.noise import gaussian_noise


noise = gaussian_noise((200, 300), 10)

Image.fromarray(noise, "L").save("images/output.jpg")