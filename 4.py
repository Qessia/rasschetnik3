import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


def main():
    df = pd.read_csv('iris1.data')
    df = df[df['class'].isin(['Iris-versicolor', 'Iris-virginica'])]  # оставляем только нужные нам классы

    x = df.drop(['width_of_sepal', 'length_of_sepal', 'width_of_petal', 'class'], axis=1)  # достаём единственный нужный нам параметр
    y = df['class']  # классы для обучения

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # делим датасет на часть для тренировки и
    # и теста в соотношении 4:1

    nb = GaussianNB()  # создаём классификатор

    nb.fit(x_train, y_train)  # обучаем классификатор

    pred = nb.predict(x_test)  # предсказываем

    acc = accuracy_score(y_test, pred) * 100  # проверяем точность предсказания
    print("accuracy:", "%.2f" % acc, '%')


if __name__ == '__main__':
    main()
