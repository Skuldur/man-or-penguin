import PIL
from PIL import Image
import glob, os

os.chdir("images/penguins")

for file in glob.glob("*.jpg"):
    basewidth = 150
    img = Image.open(file)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((150,150), PIL.Image.ANTIALIAS)
    img.save('../tmp/penguin/'+file)



