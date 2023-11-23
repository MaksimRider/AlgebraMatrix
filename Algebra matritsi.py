import numpy as np


def gaussian_elimination(matrix, b):
    n = len(matrix)

    #Створюємо матрицю [A|b]
    augmented_matrix = np.hstack((matrix, np.array(b).reshape(-1, 1)))

    #Прямий хід
    for i in range(n):
        #Підраховуємо множник для поточного рядка
        for j in range(i + 1, n):
            factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j][i:] -= factor * augmented_matrix[i][i:]

    #Зворотній хід
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i][-1] / augmented_matrix[i][i]
        for j in range(i - 1, -1, -1):
            augmented_matrix[j][-1] -= augmented_matrix[j][i] * x[i]

    return x


def kramer_method(matrix, b):
    n = len(matrix)
    det_A = np.linalg.det(matrix)

    if det_A == 0:
        print("Система рівнянь не має єдиного розв'язку (det(A) = 0).")
        return None

    solutions = []
    for i in range(n):
        #Створюємо копію матриці A і замінюємо i-тий стовпець на вектор b
        matrix_i = matrix.copy()
        matrix_i[:, i] = b

        #Знаходимо детермінант матриці заміщеної
        det_A_i = np.linalg.det(matrix_i)

        #Знаходимо розв'язок для i-тої невідомої
        xi = det_A_i / det_A
        solutions.append(xi)

    return solutions

#Отримати розмірність матриці від користувача
dimensions = input("Введіть розмірність матриці (у форматі 3-3, якщо матриця 3x3): ").split('-')
n = int(dimensions[0])
m = int(dimensions[1])

# Ввід матриці та вектора b
print("Введіть матрицю A:")
matrix = []
for i in range(n):
    row = list(map(float, input().split()))
    matrix.append(row)

print("Введіть вектор b:")
b = list(map(float, input().split()))

# Перетворення списків у масиви NumPy
matrix = np.array(matrix)
b = np.array(b)

# Запит користувача щодо методу розв'язку
method = input("Виберіть метод (Гауса або Крамера): ")

if method.lower() == "гауса":
    result = gaussian_elimination(matrix, b)
elif method.lower() == "крамера":
    result = kramer_method(matrix, b)
else:
    print("Невідомий метод")

if result is not None:
    print("Результат:")
    for x in result:
        print(x)
