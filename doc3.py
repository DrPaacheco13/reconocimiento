import cv2
import os
import dlib
import mysql.connector

# Obtén la ruta del directorio actual
current_directory = os.getcwd()

# Concatena la ruta del script con el nombre de la imagen
image_path = os.path.join(current_directory, "img/elon3.jpg")

# Conéctate a la base de datos
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="reconocimiento"
)

# Crea un cursor para ejecutar consultas
cursor = db_connection.cursor()

# Crear la tabla si no existe
# create_table_query = """
# CREATE TABLE IF NOT EXISTS encodings (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     left_coord INT,
#     top_coord INT,
#     right_coord INT,
#     bottom_coord INT
# )
# """
# cursor.execute(create_table_query)
# db_connection.commit()

# Inicializa el detector de rostros y la ventana de imagen
detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

# Procesa la imagen
img = dlib.load_rgb_image(image_path)
dets = detector(img, 1)

# Cargar coordenadas de caras previamente registradas
cursor.execute("SELECT left_coord, top_coord, right_coord, bottom_coord FROM encodings")
rows = cursor.fetchall()
detected_faces = [(row[0], row[1], row[2], row[3]) for row in rows]


# Inserta las coordenadas de las caras detectadas en la base de datos
print(dets)
for i, d in enumerate(dets):
    left_coord, top_coord, right_coord, bottom_coord = d.left(), d.top(), d.right(), d.bottom()

    # Verifica si la cara ya ha sido detectada antes
    if (left_coord, top_coord, right_coord, bottom_coord) in detected_faces:
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} - Already detected (marked as green)".format(
            i, left_coord, top_coord, right_coord, bottom_coord))
        # Marca la detección como verde en la imagen
        rect_color = (0, 255, 0)  # Verde
    else:
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} - Inserted into database (marked as red)".format(
            i, left_coord, top_coord, right_coord, bottom_coord))
        # Inserta en la base de datos
        insert_query = "INSERT INTO encodings (left_coord, top_coord, right_coord, bottom_coord) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_query, (left_coord, top_coord, right_coord, bottom_coord))
        db_connection.commit()
        # Marca la detección como roja en la imagen
        rect_color = (255, 0, 0)  # Rojo

    # Agrega las coordenadas a la lista de caras detectadas
    detected_faces.append((left_coord, top_coord, right_coord, bottom_coord))

    # Dibuja el cuadro delimitador en la imagen con el color correspondiente
    # dlib.draw_rectangle(img, d, rect_color)
    cv2.rectangle(img, (left_coord, top_coord), (right_coord, bottom_coord), rect_color, 2)


# Cierra la conexión a la base de datos
cursor.close()
db_connection.close()

img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Visualiza la imagen con las detecciones
cv2.imshow("img_bgr", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()