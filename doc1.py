import cv2
import numpy as np
import dlib
import face_recognition
import mysql.connector
import random
import string

# Función para generar un nombre aleatorio
def generate_random_name(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Conectar a la base de datos MySQL
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="reconocimiento"
)

# Crear un cursor para ejecutar consultas
cursor = db_connection.cursor()

# Cargar la imagen de entrenamiento
img_modi = cv2.imread('img/elon2.jpg')
img_modi_rgb = cv2.cvtColor(img_modi, cv2.COLOR_BGR2RGB)

# Inicializar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()



# Detectar todas las caras en la imagen
# faces = detector(img_modi_rgb)

# Detectar todas las caras en la imagen
faces = detector(img_modi_rgb)
if len(faces) > 0:
    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(img_modi, cv2.COLOR_BGR2GRAY)

    # Aplicar ecualización de histograma
    img_gray_eq = cv2.equalizeHist(img_gray)
    # Inicializar listas para almacenar información de coincidencias
    detected_faces = []
    newly_registered_faces = []

    # Comparar cada encoding con los existentes en la base de datos
    for i, face in enumerate(faces):
        # Extraer la región de la cara de la imagen
        face_image = img_gray_eq[face.top():face.bottom(), face.left():face.right()]
        # Realizar preprocesamiento adicional si es necesario
        # Por ejemplo, puedes aplicar ecualización de histograma
        # face_image = cv2.equalizeHist(face_image)

        # Detectar la codificación facial
        train_encode = face_recognition.face_encodings(face_image)
        if len(train_encode) > 0:

            # print(train_encode)

            # Convertir la codificación a una cadena para almacenarla en la base de datos
            encoded_str = ','.join(map(str, train_encode[0]))

            # Consultar todos los encodings y nombres existentes en la base de datos
            select_query = "SELECT encoding, name FROM encodings"
            cursor.execute(select_query)
            rows = cursor.fetchall()

            # Bandera para indicar si se ha encontrado una coincidencia
            match_found = False
    
            # Comparar el encoding actual con los existentes en la base de datos
            for row in rows:
                existing_encode = np.fromstring(row[0], sep=',')
                if face_recognition.compare_faces([existing_encode], train_encode)[0]:
                    print(f"RECONOCIMIENTO DETECTADO para la cara {i + 1}")
                    match_found = True
    
                    # Almacenar información de la coincidencia
                    detected_faces.append({
                        'index': i,
                        'name': row[1],
                        'color': (0, 255, 0)  # Color verde para reconocimientos detectados
                    })
                    break
                
            # Si no se encuentra una coincidencia, registrar el nuevo reconocimiento
            if not match_found:
                # Generar un nombre aleatorio y guardarlo en la base de datos
                name = generate_random_name()
                insert_query = "INSERT INTO encodings (encoding, name) VALUES (%s, %s)"
                cursor.execute(insert_query, (encoded_str, name))
                db_connection.commit()
                print(f"NUEVO RECONOCIMIENTO REGISTRADO para la cara {i + 1}")
    
                # Almacenar información del nuevo reconocimiento
                newly_registered_faces.append({
                    'index': i,
                    'name': name,
                    'color': (0, 0, 255)  # Color rojo para nuevos reconocimientos registrados
                })
        else:
            print('no encoding')
    # Dibujar bounding boxes y mostrar nombres en la imagen
    for face_info in detected_faces + newly_registered_faces:
        i = face_info['index']
        name = face_info['name']
        color = face_info['color']

        # Dibujar bounding box
        cv2.rectangle(img_modi_rgb, (faces[i].left(), faces[i].top()), (faces[i].right(), faces[i].bottom()), color, 2)

        # Mostrar el nombre generado en la parte superior de la bounding box
        cv2.putText(img_modi_rgb, name, (faces[i].left(), faces[i].top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Cerrar la conexión a la base de datos
    cursor.close()
    db_connection.close()

    # Mostrar la imagen con las bounding boxes
    cv2.imshow('Result', img_modi_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('vacio')