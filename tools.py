import streamlit as st
from PIL import Image
import numpy as np
import cv2
from fractions import Fraction
from streamlit_image_coordinates import streamlit_image_coordinates

history = [] # Tạo 1 list lưu trữ các ảnh

# Đọc ảnh
def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return image

# Hiển thị ảnh
def show_image(image, key):
    
    img_array = np.array(image) # Chuyển đổi ảnh từ PIL Image sang NumPy array
    height, width, _ = img_array.shape
    k = width/height
    value = streamlit_image_coordinates(image, key=key, width=300)
    if value is not None:
        st.write('x = {}, y = {}'.format(value['x']*width/300, value['y']*height*k/300))
    else:
        st.write('')

def show_info(image):
    
    img_array = np.array(image) # Chuyển đổi ảnh từ PIL Image sang NumPy array

    height, width, _ = img_array.shape
    st.sidebar.write(f"Kích thước: {width, height}")
    st.sidebar.write(f'Tỉ lệ: {Fraction(width,height)}')
    
# Hàm để thêm bức ảnh vào lịch sử
def add_to_history(image):
    history.append(image.copy())

# Cắt theo điểm
def crop_by_points(image, x1, y1, x2, y2):
    
    img_array = np.array(image) # Chuyển đổi ảnh từ PIL Image sang NumPy array

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Chuyển đổi x1, y1, x2, y2 sang kiểu số nguyên

    crop_by_points = img_array[y1:y2, x1:x2]

    return crop_by_points

def crop_by_ellipse(image, center_x, center_y, major_axis, minor_axis):
    
    img_array = np.array(image)
    height, width, _ = img_array.shape

    mask = np.zeros((height, width), dtype=np.uint8) # Tạo một mảng chứa điểm ảnh trắng

    cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), 0, 0, 360, 255, thickness=-1) # Vẽ ellipse trắng lên mask

    result = cv2.bitwise_and(img_array, img_array, mask=mask) # Áp dụng mask lên ảnh gốc để cắt ảnh theo ellipse

    crop_by_ellipse = Image.fromarray(result) # Chuyển đổi kết quả sang định dạng PIL Image

    return crop_by_ellipse

# Xoay ảnh
def rotate_image(image, angle):

    img_array = np.array(image)
    if angle == 90:
        rotated_image = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270:
        rotated_image = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Tính toán kích thước mới cho ảnh xoay
        height, width, _ = img_array.shape
        new_height, new_width = height, width
        
        # Tính toán ma trận biến đổi để thực hiện xoay ảnh
        center = (new_width // 2, new_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img_array = cv2.warpAffine(img_array, rotation_matrix, (new_width, new_height))

        rotated_image = Image.fromarray(rotated_img_array) # Chuyển đổi lại sang định dạng PIL Image

    return rotated_image

# Hàm zoom ảnh
def zoom_image(image, zoom_factor, coordinates):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    
    # Tính toán ma trận biến đổi để thực hiện zoom
    zoom_matrix = cv2.getRotationMatrix2D(coordinates, 0, zoom_factor)
    zoomed_img_array = cv2.warpAffine(img_array, zoom_matrix, (width, height))

    zoomed_image = Image.fromarray(zoomed_img_array) # Chuyển đổi lại sang định dạng PIL Image

    return zoomed_image

# Hàm thay đổi tỉ lệ ảnh
def resize_image(image, aspect_ratio):

    img_array = np.array(image)

    height, width, _ = img_array.shape

    aspect_fraction = Fraction(aspect_ratio) # Biến đổi tỉ lệ thành dạng phân số
    
    # Tính toán chiều rộng mới dựa trên tỉ lệ mong muốn
    new_width = int(height * aspect_fraction.numerator / aspect_fraction.denominator)
    
    # Thực hiện thay đổi kích thước
    resized_img_array = cv2.resize(img_array, (new_width, height))

    # Chuyển đổi lại sang định dạng PIL Image
    resized_image = Image.fromarray(resized_img_array)
    return resized_image

# Áp dụng công cụ và hiển thị ảnh
def apply_tools(image, selected_tools):
    
    # Thêm ảnh hiện tại vào lịch sử
    add_to_history(image)

    # Cắt ảnh
    if "Cắt ảnh" in selected_tools:
        crop_options = st.sidebar.radio("Chọn kiểu cắt ảnh", ["Cắt theo hình chữ nhật", "Cắt theo ellipse"])
        
        if crop_options == "Cắt theo hình chữ nhật":
            x1 = st.sidebar.number_input('Nhập x1:',min_value=0, value= image.width//4, help='Nhập tọa độ x của điểm trên bên trái')
            y1 = st.sidebar.number_input('Nhập y1:', min_value=0, value= image.height//4, help='Nhập tọa độ y của điểm trên bên trái')
            x2 = st.sidebar.number_input('Nhập x2:', min_value=1, value= image.width//2, help='Nhập tọa độ x của điểm dưới bên phải')
            y2 = st.sidebar.number_input('Nhập y2:', min_value=1, value= image.height//2, help='Nhập tọa độ y của điểm dưới bên phải')
            
            # Cắt ảnh theo điểm
            image = crop_by_points(image, x1, y1, x2, y2)
            
        if crop_options == "Cắt theo ellipse":
            center_x = st.sidebar.number_input("Tọa độ trung tâm X", min_value=0, max_value=image.width, value=image.width // 2)
            center_y = st.sidebar.number_input("Tọa độ trung tâm Y", min_value=0, max_value=image.width, value=image.width // 2)
            major_axis = st.sidebar.number_input("Trục Chính", min_value=1, max_value=min(image.width, image.height), value=image.width // 3)
            minor_axis = st.sidebar.number_input("Trục Phụ", min_value=1, max_value=min(image.width, image.height), value=image.width // 6)

            # Cắt ảnh theo ellipse
            image = crop_by_ellipse(image, center_x, center_y, major_axis, minor_axis)    
        
    # Xoay ảnh
    if "Xoay ảnh" in selected_tools:
        rotation_angle = st.sidebar.slider("Góc Xoay", 0, 360, 0, help='Kéo thanh trượt để chỉnh góc quay')
        image = rotate_image(image, rotation_angle)

    # Zoom ảnh
    if "Zoom" in selected_tools:
        zoom_factor = st.sidebar.slider("Tỉ lệ Zoom", 0.0, 5.0, 1.0,
                                        help='Kéo thanh trượt để chỉnh tỉ lệ phóng to hay thu nhỏ ảnh')
        x = st.sidebar.number_input('Nhập tọa độ x')
        y = st.sidebar.number_input('Nhập tọa độ y')
        coordinates = (x, y)
        image = zoom_image(image, zoom_factor, coordinates)

    if "Thay đổi tỉ lệ ảnh" in selected_tools:
        st.sidebar.subheader("Chọn Tỉ Lệ Ảnh")
        aspect_ratio = st.sidebar.selectbox("Chọn tỉ lệ ảnh",
                                            ['9/16', '2/3', '3/4', '1', '4/3', '3/2', '16/9'],
                                            help='Lựa chọn tỉ lệ mới cho bức ảnh của bạn')
        image = resize_image(image, aspect_ratio)

    return image

def main():
    st.title("Ứng Dụng Chỉnh Sửa Ảnh")

    # Chọn ảnh từ máy tính
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        col1, col2 = st.columns([1, 1])
        with col1:
            # Đọc ảnh và hiển thị ảnh gốc
            original_image = load_image(uploaded_file)
            show_image(original_image, key='original_image')

            show_info(original_image)

        st.sidebar.title("Chọn Chức Năng")

        # Tạo containers cho từng loại công cụ, hiệu ứng và điều chỉnh
        with st.sidebar.container():
            st.title("Công cụ")
            selected_tools = st.multiselect("Chọn công cụ", ["Cắt ảnh" ,"Xoay ảnh", "Zoom", "Thay đổi tỉ lệ ảnh"])

        with col2:
            # Áp dụng các công cụ và hiển thị ảnh sau khi áp dụng
            edited_image = apply_tools(original_image.copy(), selected_tools)
            if selected_tools:
                show_image(edited_image, key='edited_image')

if __name__ == "__main__":
    main()