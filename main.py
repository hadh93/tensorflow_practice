import tensorflow as tf
import math, time

# p48까지 학습한 내용

def check_tf_ver():
    print("\n\n버전 정보 확인중...\n")
    print(tf.__version__)
    # Tensorflow의 버전정보


def random_number_generation_test():
    rand = tf.random.uniform([1], 0, 1) # .uniform -> 균일분포(uniform distribution 이라는 뜻)
    # 0에서 1사이의 아무 난수나 나올 수 있다.
    # [1] => 난수의 shape 결정. 길이 1짜리 1차원 벡터라는 뜻.
    # [x,y] => shape은 x행 y열짜리 행렬

    rand_arr = tf.random.uniform([5,5],0,1)
    # 따라서 5x5 행렬이 생성되고, 0에서 1사이 숫자가 뽑힘.

    rand_normal = tf.random.normal([7], 0, 1)
    # 길이 7짜리 벡터
    # 평균 0, 표준편차 1짜리 정규분포에서 난수 뽑힘.

    print("\n\n난수 생성 시험중...\n")
    time.sleep(1)
    print(rand)
    print(rand_arr)
    print(rand_normal)

def sigmoid(x):
    # 시그모이드의 결과값(y) 의 범위는 늘 0에서 1 사이이다. (x의 값과 무관)
    # 따라서 성공을 1, 실패로 0으로 두는 binary한 결과를 표현하는데 용이하다.

    # 다만 최근에는 ReLU 함수가 더 많이 쓰이는 모양.
    return 1/ (1+math.exp(-x) )


def neuron_generation_test():

    # 입력 (x값) 이 1일때 기대출력 (y) 가 0이 되는 시나리오에서 뉴런 형성.
    # 기대출력 = 원하는 결과이다. 학습 전에는 기대출력으로 수렴하지 않는 것이 정상.
    # 학습 후 수렴한다면 Success

    x = 1
    y = 0
    w = tf.random.normal([1],0,1)
    print("\n\n뉴런 생성 시험중...\n")

    for i in range(1000):
        output = sigmoid(x*w)
        error = y-output
        w = w+x*0.1*error

        if i % 100 == 99:
            print("{}회차: {}, 오차: {}".format(i,output, error))
    print("경사 하강법 사용한 오차계산 진행 완료.")


def bias_test():
    x = 0
    y = 1
    w = tf.random.normal([1], 0, 1)
    # 경사하강법의 업데이트식이 w += w * x ... 라면, x가 0일때 학습이 멈추고 진행되지 않는다.
    # 이를 방지하기 위해, x가 0일때도 조금은 움직이게 해주는 'bias(편향)'를 설정한다.
    b = tf.random.normal([1], 0, 1)

    for i in range(1000):
        output = sigmoid(x*w+1*b)
        error = y-output
        w = w + x * 0.1 * error
        b = b + 1 * 0.1 * error

        if i % 100 == 99:
            print("{}회차: {}, 목표치: {}, 오차: {}".format(i, output, y, error))
    print("경사 하강법 사용한 오차계산 진행 완료.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    check_tf_ver()
#    time.sleep(3)
#    random_number_generation_test()
#    time.sleep(3)
#    neuron_generation_test()
    time.sleep(3)
    bias_test()
