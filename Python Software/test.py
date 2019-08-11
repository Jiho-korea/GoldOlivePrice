import tensorflow as tf
import numpy as np

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None,1])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([1,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러오는 객체를 선언합니다.
saver = tf.train.Saver()
model = tf.global_variables_initializer()

# 4가지 변수를 입력 받습니다.
mhw = float(input('최저 시급 : '))

with tf.Session() as sess:
    sess.run(model)
    save_path = "./saved.cpkt"
    saver.restore(sess, save_path)

    data = ((mhw), (0))
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:1]
    dict = sess.run(hypothesis, feed_dict={X: [x_data]})
    print(dict[0])