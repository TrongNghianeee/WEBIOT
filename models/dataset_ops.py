import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
from sklearn.model_selection import train_test_split
import sys
import os

def postpad_to(sequence, to):
    return k.preprocessing.sequence.pad_sequences(sequence, to, padding='post', truncating='post')


def map_text(X):
    if X[-1] == '/':
        X = X[:-1]
    return X


def load_and_clean(file_path=os.path.join(os.path.dirname(__file__), '../data/urldata.csv'), save_to=None):
    # Đọc tệp dữ liệu mới
    data = pd.read_csv(file_path)
    
    # Chuyển đổi cột 'label' thành nhãn số
    data['type'] = data['label'].apply(lambda x: 1 if x == 'bad' else 0)
    data = data.drop(columns=['label'])  # Bỏ cột 'label' gốc, giữ lại 'type' và 'url'
    
    # Xóa các URL trùng lặp
    data = data.drop_duplicates(subset=['url']).reset_index(drop=True)
    
    # Lưu tệp nếu cần
    if save_to is not None:
        data.to_csv(save_to, index=False)

    return data


def load_data(file_name='data/urldata.csv', split_ratio=None, random_state=42):
    # Gọi hàm load_and_clean để đọc dữ liệu
    data = load_and_clean(file_name)

    # Xáo trộn dữ liệu
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if split_ratio is None:
        data_train = data_validation = None
    else:
        data_train, data_validation = train_test_split(
            data,
            test_size=split_ratio,
            stratify=data['type'],
            shuffle=True,
            random_state=random_state
        )

    return data_train, data_validation, data


def create_dataset_preloaded(word_vectorizer, char_vectorizer, data: pd.DataFrame, one_hot=False, vec_length=200):
    assert word_vectorizer is not None or char_vectorizer is not None

    if word_vectorizer is not None:
        word_tokenizer = word_vectorizer.build_tokenizer()
        wv = tf.constant(postpad_to(
            data['url'].map(lambda url: [word_vectorizer.vocabulary_.get(a, -1)+2 for a in word_tokenizer(url)]),
            vec_length
        ), name='word')

    if char_vectorizer is not None:
        cv = tf.constant(postpad_to(char_vectorizer.texts_to_sequences(data['url']), vec_length), name='char')

    if one_hot:
        targets = tf.squeeze(tf.one_hot(data['type'], depth=2))
    else:
        targets = data['type']

    if word_vectorizer is not None:
        if char_vectorizer is not None:
            ds = tf.data.Dataset.from_tensor_slices(((wv, cv), targets))
        else:
            ds = tf.data.Dataset.from_tensor_slices((wv, targets))
    else:
        ds = tf.data.Dataset.from_tensor_slices((cv, targets))

    return ds

def create_dataset_generator(word_vectorizer, char_vectorizer, data: pd.DataFrame, one_hot=False, vec_length=200):
    assert word_vectorizer is not None or char_vectorizer is not None

    if word_vectorizer is not None:
        word_tokenizer = word_vectorizer.build_tokenizer()

    def gen():
        for row in data.iterrows():
            out_dict = dict()

            url = row[1].url
            _type = row[1].type
            if one_hot:
                target = tf.squeeze(tf.one_hot([_type], depth=2))  # Chuyển type thành one-hot encoding nếu cần
            else:
                target = tf.squeeze(_type)

            if word_vectorizer is not None:
                wv = tf.constant(postpad_to(
                    [[word_vectorizer.vocabulary_.get(a, -1) + 2 for a in word_tokenizer(url)]],  # Tokenize và biến thành số
                    vec_length
                ), name='word')
                out_dict['word'] = tf.squeeze(wv)

            if char_vectorizer is not None:
                cv = tf.constant(postpad_to(char_vectorizer.texts_to_sequences([url]), vec_length), name='char')
                out_dict['char'] = tf.squeeze(cv)

            yield out_dict, target

    # Xác định kiểu dữ liệu và kích thước của các tensor đầu ra
    output_types, output_shapes = dict(), dict()
    if word_vectorizer is not None:
        output_types['word'] = tf.float64
        output_shapes['word'] = tf.TensorShape([vec_length])
    if char_vectorizer is not None:
        output_types['char'] = tf.float64
        output_shapes['char'] = tf.TensorShape([vec_length])

    # Tạo và trả về dataset từ generator
    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(output_types, tf.int32),
        output_shapes=(output_shapes, tf.TensorShape([] if not one_hot else [2]))
    )
    return ds

