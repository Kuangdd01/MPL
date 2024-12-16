"""
Flickr or refcoco dataset regions visualization``
"""
import argparse
import json
import cv2
import base64
import os
from typing import List, Tuple
import random
parser = argparse.ArgumentParser(
    description="html generator for PG task"
)
parser.add_argument("--dataset", default="phrase", help="which dataset selected")
parser.add_argument('--mat_root', default="")
parser.add_argument("--nums",default=500 ,type=int, help="photo number to display")

flickr_root = ""

class ImageFeatures():
    def __init__(self,image_id, box_list, class_list, 
                 attribute_list) -> None:
        self.imageId = image_id
        self.box_list = box_list
        self.class_list = class_list
        self.attr = attribute_list
        assert len(self.box_list) == len(self.class_list) and len(self.class_list) == len(self.attr)

    def origin_image_base_code(self):
        img_path = flickr_root + self.imageId + ".jpg"
        try:
            img = cv2.imread(img_path)
            if img is not None:
                pass
            else:
                print("Failed to read the image. Check the file path.")
        except Exception as e:
            print(f"An error occurred: {e}")

        code = cv2.imencode('.jpg', img)[1]
        img_code = base64.b64encode(code).decode()

        return img_code 

    def cropped_image(self):
        img_path = flickr_root + self.imageId + ".jpg"
        try:
            img = cv2.imread(img_path)
            if img is not None:
                pass
            else:
                print("Failed to read the image. Check the file path.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        cropped_list = []
        cropped_code_list = []

        for box in self.box_list:
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cropped = img[y_min:y_max, x_min:x_max]
            cropped_list.append(cropped)
            img_code = cv2.imencode('.jpg', cropped)[1]
            img_code = base64.b64encode(img_code).decode()
            cropped_code_list.append(img_code)
        return cropped_list, cropped_code_list    



def generate_num_list(st, ed, num):
    if ed - st + 1 < num:
        raise ValueError("count not appropriate!")
    numbers = random.sample(range(st,ed+1), num)
    return numbers

def toy_generate(img: ImageFeatures):
    _, code_list = img.cropped_image()
    html_template = '''<img src="data:image/png;base64,{}" alt="Image">
    <figcaption>{}</figcaption></br>'''
    html_line = ""
    idx = 0
    origin_code = img.origin_image_base_code()
    html_line += html_template.format(origin_code, "origin_image:"+img.imageId)
    for code, cls, attr in zip(code_list, img.class_list, img.attr):
        html_line += html_template.format(code,str(idx) +":"+ attr + " / " + cls)
        idx += 1
    html_line += "<hr/>"
    return html_line

def main_work():
    args = parser.parse_args()
    if args.dataset == "phrase":
        file_list = os.listdir(args.mat_root)
        for file_name in file_list:
            if file_name.endswith("train_detection_dict.json"):
                with open(args.mat_root +"/" +  file_name,'r') as f:
                    train_data = json.load(f)
                    train_data_list = [(k,v) for k,v in train_data.items()]
                    random_num_list = generate_num_list(0, len(train_data_list), args.nums)
                    select_data_list = [train_data_list[idx] for idx in random_num_list]
                    feature_list = [ImageFeatures(data[0],data[1]['bboxes'],data[1]['classes'],data[1]['attrs'])
                                     for data in select_data_list]
                    total_html = ""
                    for item in feature_list:
                        total_html += toy_generate(item)
                    
                    with open("train_output.html", 'w') as h:
                        h.write(total_html)
def _vocab_test():
    args = parser.parse_args()
    import json
    from datasets._glove_tokenizer import build_glove_vocab
    glove_path = ""
    gtk = build_glove_vocab(glove_path)
    idx2word = {idx: word for word, idx in gtk.vocab.items()}
   
def html_helper_inner_batch(tag_list: List[Tuple[float,float]], image_id: List[int], out_name, row_num):
    import base64
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    def generate_html(images_with_titles):
        html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Gallery</title>
        <style>
            .image-container {
                display: inline-block;
                margin-right: 20px;
                margin-bottom: 20px;
            }
            img {
                width: 100px;
                height: auto;
            }
            .row {
                display: flex;
                flex-wrap: wrap;
            }
        </style>
    </head>
    <body>
        <div class="row">
        '''

        for index, (image_path, title) in enumerate(images_with_titles):
            base64_image = image_to_base64(image_path)
            html_content += f'''
            <div class="image-container">
                <img src="data:image/png;base64,{base64_image}" alt="null">
                <p>{str(round(title[0],3))}</p>
                <p>{str(round(title[1],3))}</p>
            </div>
            '''

            if (index + 1) % row_num == 0:
                html_content += "</hr>"
                html_content += '''
        </div>
        <div class="row">
                '''

        html_content += '''
        </div>
    </body>
    </html>
        '''
        return html_content
    image_with_titles = [(flickr_root + str(img)+ ".jpg", title) for img, title in zip(image_id, tag_list)]
    html = generate_html(images_with_titles=image_with_titles)
    with open(out_name, 'w') as f:
        f.write(html)
    return         
if __name__ == "__main__":
    main_work()
                