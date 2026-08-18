import sys, os, xml.etree.ElementTree as ET
sys.path.append('.')
from PIL import Image
from detection_yolox.yolo import YOLO

model = YOLO()
ANN_DIR  = r'C:\Users\alach\OneDrive\Desktop\modele_1\new\VOCdevkit_val\VOC2007\Annotations'
img_path = r'C:\Users\alach\OneDrive\Desktop\modele_1\new\VOCdevkit_val\VOC2007\JPEGImages\131.png'
xml_path = os.path.join(ANN_DIR, '131.xml')

print('GT boxes:')
root  = ET.parse(xml_path).getroot()
img_w = int(root.find('size/width').text)
img_h = int(root.find('size/height').text)
print(f'  Image size in XML: {img_w}x{img_h}')
for obj in root.findall('object'):
    bb = obj.find('bndbox')
    print(f'  [{bb.find("xmin").text}, {bb.find("ymin").text}, {bb.find("xmax").text}, {bb.find("ymax").text}]')

img   = Image.open(img_path).convert('RGB')
print(f'Image PIL size: {img.size}')
preds = model.detect_image(img)
print('Predictions:')
for p in preds:
    print(f'  [{int(p[0])}, {int(p[1])}, {int(p[2])}, {int(p[3])}] conf={float(p[5]):.3f}')