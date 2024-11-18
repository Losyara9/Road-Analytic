import xml.etree.ElementTree as ET

def merge_cvat_annotations(file1, file2, output_file):
    tree1 = ET.parse(file1)
    root1 = tree1.getroot()

    tree2 = ET.parse(file2)
    root2 = tree2.getroot()

    # Добавляем все <image> из второго файла в первый
    for image in root2.findall('image'):
        root1.append(image)

    tree1.write(output_file)


merge_cvat_annotations("E:\\los1\\uechebnoe\\kurs3\\detection\\annotations\\annotations1.xml",
                       "E:\\los1\\uechebnoe\\kurs3\\detection\\annotations\\annotations2.xml",
                       "E:\\los1\\uechebnoe\\kurs3\\detection\\annotations\\merged_annotations.xml")
