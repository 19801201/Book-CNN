import os
from PIL import Image
from itertools import islice




root_dir = "/raid/qzSong2_root/zhanghn/yolov4-tiny-521/"
annotations_dir = root_dir + "txt/"
image_dir = root_dir + "VOCdevkit/VOC2007/JPEGImages/"
xml_dir = root_dir + "VOCdevkit/VOC2007/Annotations/"
class_name = ['plane','ship','storage tank','baseball diamond','tennis court','swimming pool','ground track field',' harbor','bridge','large vehicle',' small vehicle','helicopter','round-about','soccer ball field','basketball court']






def txt_xml(filename):
    file_path = annotations_dir+filename
    fin = open(file_path, 'r')
    fin_lines = fin.readlines()
    image_name = filename.split('.')[0]

    img = Image.open(image_dir + image_name + ".png")
    xml_name = xml_dir + image_name + '.xml'
    with open(xml_name, 'w') as fout:
        fout.write('\t'+'<annotation>' + '\n')
        fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
        fout.write('\t' + '<filename>' + image_name + '.png' + '</filename>' + '\n')
        fout.write('\t' + '<source>' + '\n')
        fout.write('\t\t' + '<database>' + 'dota' + '</database>' + '\n')
        fout.write('\t' + '</source>' + '\n')
        fout.write('\t' + '<size>' + '\n')
        fout.write('\t\t' + '<width>' + str(img.size[0]) + '</width>' + '\n')
        fout.write('\t\t' + '<height>' + str(img.size[1]) + '</height>' + '\n')
        fout.write('\t\t' + '<depth>' + '1' + '</depth>' + '\n')
        fout.write('\t' + '</size>' + '\n')
        fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')



        line_list=[]
        for line in fin_lines:
            a = '\t' in line
            if a:
                line_content = line.replace('\t',',').replace('\n','').split(',')
            else:
                line_content = line.replace(' ', ',').replace('\n', '').split(',')
            line_list.append(line_content)
        # print(line_list)
        # exit()

        # for line in islice(input_file, 2, None):
        #     print(line)
        for i in range(2,len(line_list)):
            line_list[i]
            print(line_list)
            # cls_name = line_content[8]
            #
            # class_name = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'swimming pool',
            #               'ground track field', ' harbor', 'bridge', 'large vehicle', ' small vehicle', 'helicopter',
            #               'round-about', 'soccer ball field', 'basketball court']
            #
            # name = class_name[int(cls_name)]
            fout.write('\t' + '<object>' + '\n')
            fout.write('\t\t' + '<name>' + line_list[i][8] + '</name>' + '\n')  # true
            fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
            fout.write('\t\t' + '<truncated>' + '0' + '</truncated>' + '\n')
            fout.write('\t\t' + '<difficult>' + line_list[i][9] + '</difficult>' + '\n')  # true
            fout.write('\t\t' + '<bndbox>' + '\n')
            fout.write('\t\t\t' + '<xmin>' + line_list[i][0] + '</xmin>' + '\n')  # true
            fout.write('\t\t\t' + '<ymin>' + line_list[i][1] + '</ymin>' + '\n')  # true
            fout.write('\t\t\t' + '<xmax>' + line_list[i][4] + '</xmax>' + '\n')  # true
            fout.write('\t\t\t' + '<ymax>' + line_list[i][5] + '</ymax>' + '\n')
            fout.write('\t\t' + '</bndbox>' + '\n')
            fout.write('\t' + '</object>' + '\n')
        fin.close()
        fout.write('</annotation>')

# file_name = 'trainimg0001new.txt'
# txt_xml(file_name)
for file_name in os.listdir(annotations_dir):
    txt_xml(file_name)
