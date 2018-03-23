from PIL import Image
import draw_scanner
import sys

if __name__ == '__main__':
    image = Image.open(sys.argv[1])
    print(draw_scanner.scaner(image))