import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["1"]


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOCdevkit%s/Annotations/%s.xml' % (year, image_id), encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        # difficult = 0
        # if obj.find('difficult')!=None:
        #     difficult = obj.find('difficult').text

        cls = obj.find('clear').text
        if cls not in classes:
            continue
        # if cls not in classes or int(difficult)==1:
        #     continue
        cls_id = classes.index(cls)

        # xmlbox = obj.find('bndbox')
        b = (int(obj.find('x0').text), int(obj.find('y0').text), int(obj.find('x2').text), int(obj.find('y2').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOCdevkit%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='UTF-8')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOCdevkit%s/JPEGImages/%s.jpg' % (wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
