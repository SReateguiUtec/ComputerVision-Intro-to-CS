import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Generamos una matriz 8x8 con la imagen promedio de los digitos disponibles en este dataset, los generamos todos juntos o del digito de eleccion.
digits = datasets.load_digits()
generated_images = []

for digit in range(10):
    digit_images = digits.images[digits.target == digit]
    final_image = np.zeros((8, 8))
    for row in range(8):
        for cols in range(8):
            final_image[row][cols] = np.mean(digit_images[:, row, cols])
    generated_images.append(final_image)

final_image = np.array(generated_images)

# Hacemos un menu para dar opcion a elegir si queremos todos los digitos, solo uno o salir del menu para pasar a las siguientes preguntas.
# Para esto hacemos 3 funciones.
def mostrar_digito(digit):
    plt.imshow(final_image[digit], cmap='gray')
    plt.title("Imagen promedio del digito elegido en el menu.\n")
    plt.show()

def mostrar_digitos():
    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(final_image[i], cmap='gray')
        ax.set_title(f"{i}\n")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def menu():
    while True:
        print("Menú:")
        print("Mostrar todas los digitos, marca #1")
        print("Mostrar un digito de eleccion, marca #2")
        print("Para salir del menu, marca #3")
        print()
        print("         |       |        |       ")
        print("         |       |        |      ")
        print("         v       v        v      ")
        print()
        option = input("Ingrese su opción: ")
        print()
        if option == '#1':
            mostrar_digitos()
        elif option == '#2':
            while True:
                digit = int(input("Ingrese el dígito (0-9) --> "))
                if 0 <= digit <= 9:
                    mostrar_digito(digit)
                    break
                else:
                    print("Input invalido, debe ser un digito (0 al 9).")
        elif option == "#3":
            break
        else:
            print("Input invalido, debe ser una de las 3 opciones (#1, #2 o #3) --> ")
menu()

# Introducimos un nuevo digito convertido en 8x8 y cambiamos el valor de sus pixeles para que se pueda identificar mejor el digito.

img = cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(img, (8, 8))
trimmed_image = np.array(resized_image)

######################################################################
print("Dibujo de la imagen original, "                               #
      "casi identica a la foto tomada, para comparar visualmente:")  #
print()                                                              #
ruta_de_foto = "5.png"                                               #
imagen = Image.open(ruta_de_foto)                                    #
imagen_resized = imagen.resize((35, 8))                              #
imagen_gris = imagen_resized.convert('L')                            #
pixels = list(imagen_gris.getdata())                                 #
width, height = imagen_gris.size                                     #   #--> Representacion de la foto en el terminal dibujada con los simbolos en "caracteres"
caracteres = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']      #
                                                                     #
for i in range(height):                                              #
    print((width), end='')                                           #
    for j in range(width):                                           #
        pixel_value = pixels[i * width + j]                          #
        ascii_index = pixel_value * (len(caracteres) - 1) // 255     #
        print(caracteres[ascii_index], end='')                       #
    print()                                                          #
print()                                                              #
######################################################################

print("Arreglo de la imagen del digito a analizar: ")
print()

for i in range(8):
    for j in range(8):
        if trimmed_image[i][j] == 255:
            trimmed_image[i][j] = 0
        elif trimmed_image[i][j] >= 200 and trimmed_image[i][j] <= 255:
            trimmed_image[i][j] = 0
        elif trimmed_image[i][j] < 200:
            trimmed_image[i][j] = 255

print(trimmed_image)
print()

# Encontramos los 3 digitos mas parecidos a la imagen.
new_digit_array = trimmed_image.flatten()
distancia = []

#Calculamos las distancias
for i in range(len(digits.images)):
    digit_matrix = digits.images[i]
    digit_array = digit_matrix.flatten()
    distance = np.sqrt(np.sum((new_digit_array - digit_array) ** 2))  # Distancia euclidiana.
    distancia.append((distance, digits.target[i])) # Guardamos el indice y distancia del digito.

# Ordenamos por distancia la lista de la distancia mas cercana a la mas lejana.
distancia.sort(key=lambda x: x[0])

# Tomamos los 3 dígitos más cercanos.
closest_digits = [[distancia[0]], [distancia[1]], [distancia[2]]]

# Imprimo los targets y las distancias.
print("Los 3 digitos mas cercanos a la imagen ingresada son los siguientes")

print()
print("                     |       |        |       ")
print("                     |       |        |      ")
print("                     v       v        v      ")
print()

for distance, target in closest_digits[0]:
    print(f"Primer target mas cercano: {target} --> Distancia: {round(distance, 2)}")
    target0 = target
for distance1, target1 in closest_digits[1]:
    print(f"Segundo target mas cercano: {target1} --> Distancia: {round(distance1, 2)}")
    target1_1 = target1
for distance2, target2 in closest_digits[2]:
    print(f"Tercer target mas cercano: {target2} --> Distancia: {round(distance2, 2)}")
    target2_1 = target2
print()

# Clasificamos la imagen del digito basado en los targets mas parecidos.

targets = [distancia[0][1], distancia[1][1], distancia[2][1]]

if targets[0] == targets[1] and targets[0] == targets[2]:
    clasificacion_final = targets[0]
    print(f"Soy la inteligencia artificial version SR, y he detectado que el dígito ingresado corresponde al número {clasificacion_final} ya que todos los targets coinciden con el digito.")
elif targets[0] == targets[1] or targets[0] == targets[2]:
    clasificacion_final = targets[0]
    if targets[0] == targets[1]:
        print(
            f"Soy la inteligencia artificial version SR, y he detectado que el dígito ingresado corresponde al número {clasificacion_final} ya que el primer y segundo target coinciden.")
    else:
        print(
            f"Soy la inteligencia artificial version SR, y he detectado que el dígito ingresado corresponde al número {clasificacion_final} ya que el primer y tercer target coinciden.")
elif targets[1] == targets[2]:
    clasificacion_final = targets[1]
    print(
        f"Soy la inteligencia artificial version SR, y he detectado que el dígito ingresado corresponde al número {clasificacion_final} ya que el segundo y tercer target coinciden.")
elif targets[0] != targets[1] and targets[1] != targets[2] and targets[0] != targets[2]:
    distancia_promedio = (closest_digits[0][0] + closest_digits[1][0] + closest_digits[2][0]) / 3
    min_diferencia = float('inf')
    distancia_mas_cercana = None
    target_cercano = None
    for distancia_total, target_total in distancia:
        distancia_diferencial = abs(distancia_total - distancia_promedio)
        if distancia_diferencial < min_diferencia:
            min_diferencia = distancia_diferencial
            distancia_mas_cercana = distancia_total
            target_cercano = target_total
    print(
        f"La distancia promedio que es la mas cercana a esta distancia --> {round(distancia_mas_cercana, 2)}, le pertenece al target {target_cercano}")

# Calculamos la distancia entre el digito ingresado y las 10 imagenes promedio y imprimos la menos distancia entre la iamgen y uno de los 9 digitos.

distancia_promedio_menor = []

for i in range(len(final_image)):
    digit_matrix = final_image[i]
    digit_array = digit_matrix.flatten()
    distancia_digitos_promedio = np.sqrt(np.sum((new_digit_array - digit_array) ** 2))
    distancia_promedio_menor.append((distancia_digitos_promedio, i))  # Guardamos la distancia y el target del dígito promedio.

distancia_minima, target_promedio = min(distancia_promedio_menor, key=lambda x: x[0]) # Con el min obtenemos la distancia minima de los 10 digitos y tambien su target.

print(f"Soy la inteligencia artificial version SR y la distancia menor de la imagen del dígito con uno de los dígitos promedios es --> {distancia_minima} y le pertenece al dígito promedio con índice {target_promedio}")
