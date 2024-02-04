import math
from fractions import Fraction
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


current_page = st.sidebar.radio("", ["👋Hello", "feature", "feedback"])

if current_page == "👋Hello":
    st.title('Ứng dụng chỉnh sửa ảnh')
    st.image('image.jpg', use_column_width=True)

if current_page == "feature":
    # Đọc ảnh
    def load_image(uploaded_file):
        image = Image.open(uploaded_file)
        return image

    # Hiển thị ảnh
    def show_image(image, key):
        img_array = np.array(image)  # Chuyển đổi ảnh từ PIL Image sang NumPy array
        if len(img_array.shape) == 3:  # Nếu ảnh có 3 chiều (có kênh màu)
            height, width, _ = img_array.shape
        elif len(img_array.shape) == 2:  # Nếu ảnh chỉ có 2 chiều (ảnh xám)
            height, width = img_array.shape
        else:
            st.error("Không thể hiển thị ảnh với số chiều không hợp lệ.")
            return

        k = width / height
        value = streamlit_image_coordinates(image, key=key, width=300)
        if value is not None:
            st.write('x = {}, y = {}'.format(value['x'] * width / 300, value['y'] * height * k / 300))
        else:
            st.write('')

            
    # Hiển thị một số thông tin ảnh
    def show_info(image):
        
        img_array = np.array(image) # Chuyển đổi ảnh từ PIL Image sang NumPy array

        height, width, _ = img_array.shape
        st.sidebar.write(f"Kích thước: {width, height}")
        st.sidebar.write(f'Tỉ lệ: {Fraction(width,height)}')
    
    # Code phần công cụ chỉnh ảnh
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

    # Chức năng xoay ảnh
    def rotate_image(image, angle):

        img_array = np.array(image)
        if angle == 90:
            rotated_image = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 270:
            rotated_image = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Tính toán kích thước mới cho ảnh xoay
            img_array = np.array(image)
            height, width, _ = img_array.shape
            
            # Tính toán kích thước mới dựa trên hình chữ nhật bao quanh ảnh xoay
            new_height = int(abs(width * math.sin(math.radians(angle))) + abs(height * math.cos(math.radians(angle))))
            new_width = int(abs(height * math.sin(math.radians(angle))) + abs(width * math.cos(math.radians(angle))))
            
            # Tính toán ma trận biến đổi để thực hiện xoay ảnh
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            
            # Thay đổi kích thước của ảnh xoay để không bị cắt góc
            rotation_matrix[0, 2] += (new_width - width) // 2
            rotation_matrix[1, 2] += (new_height - height) // 2
            
            rotated_img_array = cv2.warpAffine(img_array, rotation_matrix, (new_width, new_height))
            
            rotated_image = Image.fromarray(rotated_img_array)  # Chuyển đổi lại sang định dạng PIL Image

        return rotated_image

    # Chức năng zoom ảnh
    def zoom_image(image, zoom_factor, coordinates):
        img_array = np.array(image)
        height, width, _ = img_array.shape
        
        # Tính toán ma trận biến đổi để thực hiện zoom
        zoom_matrix = cv2.getRotationMatrix2D(coordinates, 0, zoom_factor)
        zoomed_img_array = cv2.warpAffine(img_array, zoom_matrix, (width, height))

        zoomed_image = Image.fromarray(zoomed_img_array) # Chuyển đổi lại sang định dạng PIL Image

        return zoomed_image

    # Chức năng thay đổi tỉ lệ ảnh
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

    # Chức năng lật ảnh
    def flip_bottom(image, dimension):
        img_array = np.array(image)
        flipped_image = cv2.flip(img_array, dimension)
        return Image.fromarray(flipped_image)
    
    # Code phần điều chỉnh ảnh                                  
    # Độ sáng
    def adjust_image_gamma_lookuptable(image, gamma=1.0):
        # Chuyển đổi image thành mảng NumPy
        img_array = np.array(image)

        # Tính toán bảng lookup
        table = np.array([(1.0 - (1.0 - i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # Áp dụng gamma correction sử dụng bảng lookup
        adjusted_img_array = cv2.LUT(img_array, table)

        # Chuyển đổi lại sang định dạng PIL Image
        adjusted_image = Image.fromarray(adjusted_img_array)

        return adjusted_image

    # Độ ấm
    def adjust_temperature(image, factor):
        
        
        enhancer = ImageEnhance.Color(image) # Tạo một đối tượng Enhancer cho Temperature
        image_with_temperature = enhancer.enhance(factor) # Áp dụng hiệu ứng Temperature
        
        return (image_with_temperature)
        
    # Độ tương phản
    def adjust_contrast(image, contrast_factor):
        enhancer = ImageEnhance.Contrast(image)
        adjusted_image = enhancer.enhance(contrast_factor)
        return adjusted_image
    
    # Độ bão hòa
    def adjust_saturation(image, saturation_factor):
        enhancer = ImageEnhance.Color(image)
        adjusted_image = enhancer.enhance(saturation_factor)
        return adjusted_image
    
    # Hiệu ứng ảnh xám
    def gray_Image(image):
        converted_img = np.array(image.convert('RGB'))
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        return gray_scale
    
    # Ảnh đen trắng
    def Black_and_White(image, slider):
        converted_img = np.array(image.convert('RGB'))
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
        return blackAndWhiteImage
    
    # Phác thảo bút chì
    def Pencil_Sketch(image, slider):
        converted_img = np.array(image.convert('RGB')) 
        gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray_scale
        blur_image = cv2.GaussianBlur(inv_gray, (slider,slider), 0, 0)
        sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
        return sketch
    
    # Làm mờ ảnh
    def Blur_Effect(image, slider):
        converted_img = np.array(image.convert('RGB'))
        # converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
        blur_image = cv2.GaussianBlur(converted_img, (slider,slider), 0, 0)
        return blur_image
    
    # Làm mịn ảnh
    def Smooth_Effect(image, slider):
        converted_img = np.array(image.convert('RGB'))
        kernel = np.ones((slider, slider), np.float32) / (slider ** 2) # Tạo kernel làm mịn
        smoothed_image = cv2.filter2D(converted_img, -1, kernel) # Làm mịn ảnh bằng filter2D
        return smoothed_image
    
    def process_logo_contour(image):
        convert_image = (image.convert('L'))
        threshold = 50
        convert_image = convert_image.point(lambda x: 255 if x > threshold else 0)

        # Áp dụng bộ lọc Contour
        convert_image = convert_image.filter(ImageFilter.CONTOUR)
        return convert_image
    
    # Hàm chuyển đổi từ mã màu hex sang tuple BGR
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Chèn chữ lên ảnh
    def add_text_to_image(image, text, position, font_scale, font_color, font_thickness):
        img_array = np.array(image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_array, text, position, font, font_scale, hex_to_bgr(font_color), font_thickness, cv2.LINE_AA)

        return Image.fromarray(img_array)

    # Xóa phông
    def image_bg_remover(image):
        fixed = remove(image)
        return fixed
    
    def convert_image(img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img) # Nếu img là NumPy array, chuyển đổi thành đối tượng Image
            
        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        return byte_im

    # Áp dụng công cụ và hiển thị ảnh
    def apply_tools(image, selected_tools):
    
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
            
        if "Lật ảnh" in selected_tools:
            options=st.sidebar.selectbox("Chọn kiểu lật", ["Chiều dọc", "Chiều ngang"])
            d = 0
            if "Chiều ngang" in options:
                d = 1
            image = flip_bottom(image, d)
            
        if "Độ sáng" in selected_tools:
            gamma_value = st.sidebar.slider("Chọn độ sáng", 0.0, 10.0, 5.0, 0.5, help="Kéo thanh trượt để chỉnh độ sáng")
            image = adjust_image_gamma_lookuptable(image, gamma_value/5)
            
        if "Độ ấm" in selected_tools:
            factor = st.sidebar.slider("Chon độ ấm", 0.0, 10.0, 5.0, 0.5, help="Kéo thanh trượt để chỉnh độ ấm")
            image = adjust_temperature(image, factor/5)
        
        if "Độ tương phản" in selected_tools:
            contrast_factor = st.sidebar.slider("Chọn Độ Tương Phản", 0.0, 10.0, 5.0, 0.5, help="Kéo thanh trượt để chỉnh độ tương phản")
            image = adjust_contrast(image, contrast_factor/5)
        
        if "Độ bão hòa" in selected_tools:
            saturation_factor = st.sidebar.slider("Chọn độ bão hòa", 0.0, 10.0, 5.0, 0.5, help="Kéo thanh trượt để chỉnh độ bão hòa")
            image = adjust_saturation(image, saturation_factor/5)
            
        if "Original" in selected_tools:
            image = image
        
        if "Gray Image" in selected_tools:
            image = gray_Image(image)
            
        if "Black and White" in selected_tools:
            slider = st.sidebar.slider("Điều chỉnh cường độ", 1, 255, 127, 1)
            image = Black_and_White(image, slider)
        
        if "Pencil Sketch" in selected_tools:
            slider = st.sidebar.slider("Điều chỉnh cường độ", 5, 255, 125, 2)
            image = Pencil_Sketch(image, slider)
            
        if "Blur Effect" in selected_tools:
            slider = st.sidebar.slider("Điều chỉnh độ mờ", 5, 125, 15, 2)
            image = Blur_Effect(image, slider)
            
        if "Smooth Effect" in selected_tools:
            slider = st.sidebar.slider("Điều chỉnh độ mịn", 1, 10, 5, 1)
            image = Smooth_Effect(image, slider)
        
        if "process_logo_contour" in selected_tools:
            image = process_logo_contour(image)
        
        if "Chèn chữ" in selected_tools:
            text_to_add = st.sidebar.text_input("Nhập văn bản", value="HIT")
            position_x = st.sidebar.slider("Vị trí X", 0, image.width, image.width // 2)
            position_y = st.sidebar.slider("Vị trí Y", 0, image.height, image.height // 2)

            font_scale = st.sidebar.slider("Cỡ chữ", 1.0, 40.0, 8.0, 1.0)
            font_color = st.sidebar.color_picker("Màu chữ", "#FF5733")
            font_thickness = st.sidebar.slider("Độ đậm của chữ", 1, 10, 2)

            image = add_text_to_image(image, text_to_add, (position_x, position_y), font_scale, font_color, font_thickness)
        
        if "Xóa phông" in selected_tools:
            image = image_bg_remover(image)
            
        return image

    def main():
        st.title("Ứng Dụng Chỉnh Sửa Ảnh")
        uploaded_file = None
        # Chọn ảnh từ máy tính
        image_source = st.radio('Chọn nguồn ảnh', ['Webcam', 'Chọn ảnh từ máy'])
        if image_source == 'Webcam':
            uploaded_file = st.camera_input('Chụp ảnh từ webcam')
        else:
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
                selected_functions = st.multiselect("Chọn chức năng", ["Công cụ", "Điều chỉnh", "Hiệu ứng", "Chữ", "Xóa phông"])
                selected_tools_1 = []
                selected_tools_2 = []
                selected_tools_3 = []
                selected_tools_4 = []
                selected_tools_5 = []
                selected_tools_6 = []
                
                if "Công cụ" in selected_functions:
                    selected_tools_1 = st.multiselect("Chọn công cụ", ["Cắt ảnh", "Xoay ảnh", "Zoom", "Thay đổi tỉ lệ ảnh", "Lật ảnh"])
                
                if "Điều chỉnh" in selected_functions:
                    selected_tools_2 = st.multiselect("Điều chỉnh", ["Độ sáng", "Độ ấm", "Độ tương phản", "Độ bão hòa"])
                
                if "Hiệu ứng" in selected_functions:
                    selected_tools_3 = st.radio("filters", ["Original", "Gray Image", "Black and White", "Pencil Sketch", "Blur Effect", "Smooth Effect", "process_logo_contour"])
                
                if "Chữ" in selected_functions:
                    selected_tools_4 = st.radio("", ["Chèn chữ"])
                    
                if "Xóa phông" in selected_functions:
                    selected_tools_6 = st.radio("", ["Xóa phông"])

                # Gộp danh sách công cụ đã chọn
                selected_tools = selected_tools_1 + selected_tools_2 + [selected_tools_3] + [selected_tools_4] + [selected_tools_5] + [selected_tools_6]

            with col2:
                # Áp dụng các công cụ và hiển thị ảnh sau khi áp dụng
                edited_image = apply_tools(original_image.copy(), selected_tools)
                if selected_tools:
                    show_image(edited_image, key='edited_image')
                    
            st.sidebar.download_button("Tải ảnh", convert_image(edited_image), "new_image.png", "image/png")
                    
    if __name__ == "__main__":
        main()

if current_page == "feedback":

    def feed_back():
        # Đọc dữ liệu hiện tại từ tệp văn bản (nếu có)
        try:
            feedback_data = pd.read_csv('feedback_data.csv')
        except FileNotFoundError:
            # Nếu tệp không tồn tại, tạo DataFrame mới
            feedback_data = pd.DataFrame(
                columns=['Name', 'Age', 'Address', 'PhoneNumber', 'Link FB', 'Rating', 'Feedback'])
            feedback_data.to_csv('feedback_data.csv', index=False)

        st.subheader('Xin hãy giúp chúng tôi cải thiện!')
        with st.form(key='columns_in_form', clear_on_submit=True):
            name = st.text_input("Tên đầy đủ", help='Điền họ và tên của bạn')
            age = st.text_input('Tuổi', help='Điền tuổi của bạn')
            phonenumber = st.text_input('Số điện thoại', help='Điền số điện thoại')
            address = st.text_input('Địa chỉ', help='Cho xin cái tọa độ))')
            linkFB = st.text_input('Link FB', help='Cho xin in4 đê :>')
            rating = st.slider("Đáng giá app", min_value=1, max_value=10, value=1,
                               help='Kéo thanh trượt để xếp hạng ứng dụng. Đây là thang đánh giá từ 1 đến 10 trong đó 10 là mức đánh giá cao nhất')
            text = st.text_input(label='Xin hãy để lại thông tin phản hồi của bạn tại đây')
            submitted = st.form_submit_button('Gửi')
            if submitted:
                st.write('Cảm ơn đã để lại đánh giá!')

                # Lưu thông tin đánh giá vào DataFrame
                new_feedback = pd.DataFrame(
                    {'Name': [name], 'Age': [age], 'PhoneNumber': [phonenumber], 'Address': [address], 'Link FB': [linkFB],
                     'Rating': [rating], 'Feedback': [text]})
                feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)

                # Lưu DataFrame vào tệp văn bản
                feedback_data.to_csv('feedback_data.csv', index=False)

    if __name__ == "__main__":
        feed_back()
