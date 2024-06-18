from pyvirtualdisplay import Display
from PIL import Image, ImageDraw, ImageFont

# ASCII Art
ascii_art = """
   _____
  /     \\
 | () () |
  \\  ^  /
   |||||
   |||||
"""

# 创建虚拟显示器
display = Display(visible=0, size=(800, 600))
display.start()

# 创建一张空白图片
img = Image.new('RGB', (800, 600), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# 设置字体和大小
# Note: Make sure to have a valid font path. This example uses a common Linux font path.
font_path = "/Users/xnpeng/Downloads/dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(font_path, 24)

# 在图片上绘制ASCII艺术
draw.text((10, 10), ascii_art, font=font, fill=(0, 0, 0))

# 显示图片
img.show()

# 停止虚拟显示器
display.stop()
