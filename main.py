import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageFilter,ImageEnhance

img = cv2.imread(r"1.jpg", -1)
resized_image1 = cv2.resize(img, (600, 900))

# Độ phơi sáng thấp (ảnh tối)
# def adjust_image_gamma(image, gamma = 1.0):
#   image = np.power(image, gamma)
#   max_val = np.max(image.ravel())
#   image = image/max_val * 255
#   image = image.astype(np.uint8)
#   return image
# low_adjusted = adjust_image_gamma(img, 0.45)
# cv2.imshow("Hien thi", low_adjusted)
# k = cv2.waitKey()


# def adjust_image_gamma_lookuptable(image, gamma=1.0):
#     # build a lookup table mapping the pixel values [0, 255] to
#     # their adjusted gamma values
#     table = np.array([((i / 255.0) ** gamma) * 255
#         for i in np.arange(0, 256)]).astype("uint8")
#     # apply gamma correction using the lookup table
#     return cv2.LUT(image, table)

# low_adjusted = adjust_image_gamma_lookuptable(img, 0.45)
# cv2.imshow("Hien thi", low_adjusted[:,:,::-1])
# k = cv2.waitKey()



# Tìm các cạnh bằng phương pháp Canny
# edges = cv2.Canny(img, 100, 200)
# cv2.imshow('image', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Vẽ countour của ảnh
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
# contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
# for contour in contours:
#     cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Nhận diện gương mặt bằng OpenCV
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags=cv2.CASCADE_SCALE_IMAGE
# )
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # cv2.imwrite('queen-fd.jpg', img)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Làm tối 4 góc (Vignette)
# rows, cols = img.shape[:2]
# # generating vignette mask using Gaussian kernels
# kernel_x = cv2.getGaussianKernel(cols, 200)
# kernel_y = cv2.getGaussianKernel(rows, 200)
# kernel = kernel_y * kernel_x.T
# mask = 255 * kernel / np.linalg.norm(kernel)
# output = np.copy(img)
# # applying the mask to each channel in the input image
# for i in range(3):
#     output[:, :, i] = output[:, :, i] * mask

# # cv2.imwrite('rick-morty-vig.png', output)
# cv2.imshow('image', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Làm mờ (Gaussian blur)
# PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
# img = Image.open("1.jpg")
# new_img = img.filter(ImageFilter.GaussianBlur(radius=20))
# new_img.show("1.jpg")



# Thay đổi độ sáng (Brightness)
# PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
# img = Image.open("1.jpg")
# enhancer = ImageEnhance.Brightness(img)
# new_img = enhancer.enhance(1.8)
# # Darker
# # new_img = enhancer.enhance(0.8)
# new_img.show("12.jpg")


# Thay đổi độ tương phản (Contrast)
# PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
# img = Image.open("1.jpg")
# # Enhance constrast
# enhancer = ImageEnhance.Contrast(img)
# for i in range(1, 8):
#     factor = i / 4.0
#     new_img = enhancer.enhance(factor)
#     new_img.show('12%s.jpg' % i)


# Chuyển ảnh màu sang Halftone
# from PIL import Image
# img = Image.open("1.jpg")
# # black and white image
# new_img = img.convert('1')
# new_img.show('14.jpg')


# Chuyển ảnh màu sang Grayscale
# img = Image.open("1.jpg")
# # If you want a greyscale image, simply convert it to the L (Luminance) mode:
# new_img = img.convert('L')
# # new_img.save('15.jpg')
# new_img.show('15.jpg')



#sketch img
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# invert  =cv2.bitwise_not(gray_img)
# blur = cv2.GaussianBlur(invert, (21,21),0)
# invertedblur = cv2.bitwise_not(blur) 
# sketch = cv2.divide(gray_img, invertedblur, scale=256.0)
# cv2.imshow('image', sketch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# làm nét ảnh
# # Khai báo đường dẫn filehinh
# filehinh = r"D:\Python\Project_Python_Private_2023\8.jpg"
# # Đọc ảnh màu dùng thư viện opencv
# img = cv2.imread(filehinh,cv2.IMREAD_COLOR)
# # Đọc ảnh màu dùng thư viện PIL . Ảnh này cho phép xử lí các tác vụ tính toán
# # và xử lí ảnh thay vì dùng thư viện opencv
# imgPIL = Image.open(filehinh)
# # Tạo 1 ảnh có cùng kích thước với Mode và imgPIL
# # Ảnh này  dùng để chứa kết quả làm sắc nét ảnh
# Hinhlamnet = Image.new(imgPIL.mode,imgPIL.size)
# # Lấy kích thước ảnh 
# w = Hinhlamnet.size[0]
# h = Hinhlamnet.size[1]
# # Mỗi ảnh là 1 ma trận 2 chiều nên dùng 2 vòng for để quét tất cả các pixel có trong ảnh
# matrix = np.array([[0, -1, 0], [-1, 4 , -1],[0, -1, 0]])
# for x in range(1,w -1):
#     for y in range(1,h -1):
#         # lấy giá trị điểm ảnh tại(x,y)
#         # Các biến này dùng để chứa giá trị cộng dồn của các điểm ảnh
#         # nằm trong mặt nạ
#         Rs = 0
#         Gs = 0
#         Bs = 0
#         Rs1 = 0
#         Gs1 = 0
#         Bs1 = 9
#         # Tiến hành quét các điểm trong mặt nạ
#         for i in range (x -1, x+2):
#             for j in range (y -1, y+2):
#                 # lấy thông tin màu R-G-B trong mặt nạ tại vị trí i,j
#                 R,G,B = imgPIL.getpixel((i,j))
#                 # Nhân tích chập các R-G-B tương ứng
#                 Rs += R *matrix[i -x + 1,j -y +1]
#                 Gs += G *matrix[i -x + 1,j -y +1]
#                 Bs += B *matrix[i -x + 1,j -y +1]
#                 Rs1,Gs1,Bs1 = imgPIL.getpixel((x,y))
#                 DKR = Rs1+Rs
#                 DKG = Gs1+Gs
#                 DKB = Bs1+Bs
#                 #_________________________________________
#                 if(Rs1 < 0):
#                     DKR = 0 
#                 else:
#                     DKR = 255
#                 if(Gs1 < 0):
#                     DKG = 0 
#                 else:
#                     DKG = 255
#                 if(Bs1 < 0):
#                     DKB = 0 
#                 else:
#                     DKB = 255
#                 Hinhlamnet.putpixel((x,y),(Bs1,Gs1,Rs1))
# # Chuyển ảnh PIL sang opencv để hiển thị lên TV opencv
# HINHLAMSACNET = np.array(Hinhlamnet)
# # Cho hiển thị ảnh lên TV openCV
# cv2.imshow('Hinh RGB',img)
# cv2.imshow('Hinh lam sac net',HINHLAMSACNET)
# # Nhấn phím bất kì để đóng cửa sổ hiển thị
# cv2.waitKey(0)
# # Giải phóng bộ nhớ đã cấp phát cho cửa sổ hiển thị
# cv2.destroyAllWindows()