import streamlit as st
import pandas as pd

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