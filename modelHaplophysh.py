import tensorflow as tf
import numpy as np
from models import dataset_ops, vectorization_ops

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('full_convolution_combined.keras')

# Tạo dữ liệu huấn luyện để khởi tạo char_vectorizer
data_train, _, _ = dataset_ops.load_data(split_ratio=0.3, random_state=42)
word_vectorizer = vectorization_ops.create_word_vectorizer(data_train['url'])
char_vectorizer = vectorization_ops.create_char_vectorizer(data_train['url'])

# URL cần kiểm tra
new_url = "http://www.google.com"

# Vector hóa URL
url_word_vector = word_vectorizer.transform([new_url]).toarray()
url_char_vector = char_vectorizer.texts_to_sequences([new_url])

# Pad URL để phù hợp với độ dài đầu vào của mô hình
input_length = 200
if url_word_vector.shape[1] < input_length:
    word_padding = np.zeros((1, input_length - url_word_vector.shape[1]))
    url_word_vector_padded = np.hstack((url_word_vector, word_padding))
else:
    url_word_vector_padded = url_word_vector[:, :input_length]
    
url_char_vector_padded = tf.keras.preprocessing.sequence.pad_sequences(
    url_char_vector, maxlen=input_length, padding='post'
)

# Dự đoán xác suất
probability = model.predict(
    (url_word_vector_padded, url_char_vector_padded)
)[0][0]

# Sử dụng ngưỡng để phân loại
best_threshold = np.load("fpr_tpr/best_threshold_fccombi.npy") #0.19

print(f'Ngưỡng: {best_threshold}')

if probability >= best_threshold:
    print(f'URL "{new_url}" là ĐỘC HẠI với xác suất {probability:.2f}')
else:
    print(f'URL "{new_url}" là AN TOÀN với xác suất {probability:.2f}')
