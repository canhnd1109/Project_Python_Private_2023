import sys
import numpy as np
import cv2
from rembg import remove
from tkinter import filedialog
from tkinter import Tk
import tkinter as tk
from PIL import Image, ImageTk
import requests

# Tao hop thoai chon file
def choose_file():
    root = Tk()
    root.withdraw() # An cua so chinh cua tkinter

    file_path = filedialog.askopenfilename(title="Chon mot file.")

    if file_path:
        print("Da chon file: ", file_path)
        return file_path
    else:
        print("Khong co file nao duoc chon.")

#Viet chu len anh
def write_text_on_img(img):
    text = input("Van ban can nhap: ")
    font_scale = float(input("Co chu: "))
    font_thickness = int(input("Do day: "))
    font_color = tuple(map(int, input("Mau sac (BGR format, vi du 255 0 0 la xanh nuoc bien): ").split()))
    font_face = cv2.FONT_HERSHEY_COMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, font_thickness)
    while True:
        print("Vi tri cua chu: ")
        print("1. Co san")
        print("2. Tuy chon")
        choose = int(input("Nhap lua chon: "))
        if choose == 1 or choose == 2:
            break
        else: print("Vui long nhap lai.")
    if choose == 1:
        print("Chon cac loai vi tri.")
        print("1. Giua anh.")
        print("2. Phai anh.")
        print("3. Trai anh.")
        print("4. Goc tren phai.")
        print("5. Goc tren trai.")
        print("6. Goc duoi phai.")
        print("7. Goc duoi trai.")
        while True:
            c = int(input("Nhap lua chon: "))
            if 1 <= c <= 7:
                break
            else: print("Vui long nhap lai")
        if c == 1:
            x = (width - text_width) // 2
            y = (height - text_height) // 2
        elif c == 2:
            x = 0
            y = (height - text_height) // 2
        elif c == 3:
            x = width - text_width
            y = (height - text_height) // 2
        elif c == 4:
            x = 0
            y = text_height
        elif c == 5:
            x = width - text_width
            y = text_height
        elif c == 6:
            x = 0
            y = height - text_height
        elif c == 7:
            x = width - text_width
            y = height - text_height
    elif choose == 2:
        x = int(input("Nhap x: "))
        y = int(input("Nhap y: "))

    cv2.putText(img, text, (x, y), font_face, font_scale, font_color, font_thickness, cv2.LINE_AA)


# Ghep, chen anh
def collage_insert_img(img):
    overlay_img = cv2.imread('dog.jpg')
    overlay_img = cv2.resize(overlay_img, (width, height))
    alpha = 0.5
    beta = 1.0 - alpha
    blended = cv2.addWeighted(img, alpha, overlay_img, beta, 0)
    cv2.imshow("Anh da chen", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Ham xu ly su kien chuot
def event_mouse(event, x, y, flags, param):
    global points, color, thickness, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
            if len(points) > 1:
                cv2.line(img, points[-2], points[-1], color, thickness)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Ve len anh
def draw_on_img(img):
    global points, color, thickness, drawing
    drawing = False
    points = []
    color = tuple(map(int, input("Mau sac (BGR format, vi du 255 0 0 la xanh nuoc bien): ").split()))
    thickness = int(input("Nhap do day: "))
    cv2.namedWindow("Ve len anh")
    cv2.setMouseCallback("Ve len anh", event_mouse)
    while True:
        temp_img = img.copy()
        if len(points) > 1:
            cv2.polylines(temp_img, [np.array(points)], isClosed=False, color=color, thickness=thickness)
        cv2.imshow("Ve len anh", temp_img)
        #Dung phim esc de tat
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

# Xoa phong
def remove_background():
    inp = file_path
    out = 'xoa_phong.png'

    input = Image.open(inp)
    output = remove(input)
    output.save(out)

    img = cv2.imread(out)

    cv2.imshow('Anh da xoa phong', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(file_path, width, height):
    original_image = cv2.imread(file_path)
    resized_image = cv2.resize(original_image, (width, height))
    return resized_image

# def download_file(url, destination):
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(destination, 'wb') as file:
#             file.write(response.content)
#         print(f"File da duoc tai va luu tai: {destination}")
#     else:
#         print(f"Loi {response.status_code}: Khong the tai file tu {url}")


# file_url = "https://examples.com"
# destination_path = "D:\BTL-HIT-Python"

# download_file(file_url, destination_path)

file_path = choose_file()

print("Chon anh can chinh sua: ")
# root = Tk()
# root.withdraw()

# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

img = cv2.imread(file_path)
# img = resize_image(file_path, screen_width - 150, screen_height - 150)

height, width, channels = img.shape

while True:
    print("1. Viet chu len anh.")
    print("2. Chen anh.")
    print("3. Ve len anh.")
    print("4. Xoa phong anh.")
    print("0. Ket thuc chuong trinh.")
    c = int(input("Nhap: "))
    if 0 <= c <= 4:
        break
    else: print("Yeu cau nhap lai.")
if c == 1:
    write_text_on_img(img)
    cv2.imshow("Anh meo", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif c == 2:
    collage_insert_img(img)
elif c == 3:
    draw_on_img(img)
elif c == 4:
    remove_background()
else :
    print("Chuong trinh da ket thuc.")
    sys.exit
