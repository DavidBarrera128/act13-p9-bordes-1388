import cv2

# 1. Cambia esta variable por la ruta y nombre real donde guardaste la imagen del gorila
ruta_imagen = "Gorila.jpg"  # O "Gorila.jpg" si está junto al script

# Cargar imagen
imagen = cv2.imread(ruta_imagen)

# Comprobar si la imagen se cargó correctamente
if imagen is None:
    print(f"Error: no se pudo cargar la imagen desde '{ruta_imagen}'.")
    print("Verifica que el nombre del archivo y la ruta sean correctos.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a imagen binaria mediante umbral
# (Se usa 100 en lugar de 127 para adaptarse a los tonos oscuros del gorila)
_, binaria = cv2.threshold(
    gris,
    100,
    255,
    cv2.THRESH_BINARY
)

# Detectar contornos
contornos, jerarquia = cv2.findContours(
    binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Dibujar los contornos en una copia
resultado = imagen.copy()

cv2.drawContours(
    resultado,
    contornos,
    -1,
    (0, 255, 0),  # Color verde
    2
)

# Mostrar resultados
cv2.imshow("original NC = 1388", imagen)
cv2.imshow("binaria NC = 1388", binaria)
cv2.imshow("Contornos NC = 1388", resultado)

# Guardar resultado
ruta_salida = "../resultados/Gorila_contornos.jpg"
cv2.imwrite(ruta_salida, resultado)

print("Cantidad de contornos encontrados:", len(contornos))
print(f"Resultado guardado en {ruta_salida}")

# Esperar a que se presione cualquier tecla para cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()