import cv2
import numpy as np

def stackImages(imgArray, scale, labels=[]):
    """
    Apilar imágenes en una cuadrícula con etiquetas mejoradas
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    
    if not rowsAvailable:
        imgArray = [imgArray]
        rows = 1
    
    # Obtener dimensiones de la primera imagen
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    
    # Redimensionar todas las imágenes
    for x in range(rows):
        for y in range(cols):
            if imgArray[x][y] is not None:
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                # Convertir imágenes en escala de grises a BGR
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
    
    # Crear imagen en blanco para rellenar espacios vacíos
    imageBlank = np.zeros((int(height * scale), int(width * scale), 3), np.uint8)
    
    # Apilar horizontalmente
    hor = []
    for x in range(rows):
        row_images = []
        for y in range(cols):
            if imgArray[x][y] is not None:
                row_images.append(imgArray[x][y])
            else:
                row_images.append(imageBlank)
        hor.append(np.hstack(row_images))
    
    # Apilar verticalmente
    ver = np.vstack(hor)
    
    # Agregar etiquetas si se proporcionan
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        
        for d in range(rows):
            for c in range(cols):
                # Obtener texto de la etiqueta
                if isinstance(labels[d], list) and c < len(labels[d]):
                    text = str(labels[d][c])
                else:
                    text = str(labels[d]) if d < len(labels) else ""
                
                if text:
                    # Calcular posición del texto
                    x_pos = c * eachImgWidth
                    y_pos = d * eachImgHeight
                    
                    # Fondo para el texto con transparencia
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Rectángulo de fondo
                    cv2.rectangle(ver, 
                                (x_pos, y_pos), 
                                (x_pos + text_size[0] + 20, y_pos + 30),
                                (50, 50, 50), cv2.FILLED)
                    
                    # Borde del rectángulo
                    cv2.rectangle(ver, 
                                (x_pos, y_pos), 
                                (x_pos + text_size[0] + 20, y_pos + 30),
                                (200, 200, 200), 2)
                    
                    # Texto
                    cv2.putText(ver, text, 
                              (x_pos + 10, y_pos + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return ver

def reorder(myPoints):
    """
    Reordenar puntos para transformación de perspectiva
    """
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    
    # Sumar coordenadas para encontrar esquinas
    add = myPoints.sum(1)
    
    # Esquina superior izquierda (suma mínima)
    myPointsNew[0] = myPoints[np.argmin(add)]
    # Esquina inferior derecha (suma máxima)
    myPointsNew[3] = myPoints[np.argmax(add)]
    
    # Diferencia de coordenadas para encontrar otras esquinas
    diff = np.diff(myPoints, axis=1)
    # Esquina superior derecha (diferencia mínima)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    # Esquina inferior izquierda (diferencia máxima)
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    return myPointsNew

def biggestContour(contours):
    """
    Encontrar el contorno más grande que sea rectangular
    """
    biggest = np.array([])
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrar contornos muy pequeños
        if area > 5000:
            # Calcular perímetro
            peri = cv2.arcLength(contour, True)
            # Aproximar contorno a polígono
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Verificar que sea un cuadrilátero y que sea el más grande
            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area
    
    return biggest, max_area

def drawRectangle(img, biggest, thickness):
    """
    Dibujar rectángulo conectando los 4 puntos del contorno
    """
    if biggest.size != 0:
        # Puntos del rectángulo
        points = biggest.reshape(4, 2)
        
        # Dibujar líneas conectando los puntos
        cv2.line(img, tuple(points[0]), tuple(points[1]), (0, 255, 0), thickness)
        cv2.line(img, tuple(points[1]), tuple(points[3]), (0, 255, 0), thickness)
        cv2.line(img, tuple(points[3]), tuple(points[2]), (0, 255, 0), thickness)
        cv2.line(img, tuple(points[2]), tuple(points[0]), (0, 255, 0), thickness)
        
        # Dibujar círculos en las esquinas
        for point in points:
            cv2.circle(img, tuple(point), 8, (0, 0, 255), cv2.FILLED)
    
    return img

def nothing(x):
    """
    Función vacía para callbacks de trackbars
    """
    pass

def initializeTrackbars(initialTracbarVals=[200, 200]):
    """
    Inicializar trackbars para control de umbralización
    """
    window_name = "Controles de Umbralización"
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 400, 200)
    
    # Crear trackbars con valores iniciales mejorados
    cv2.createTrackbar("Umbral Inferior", window_name, initialTracbarVals[0], 255, nothing)
    cv2.createTrackbar("Umbral Superior", window_name, initialTracbarVals[1], 255, nothing)
    
    # Agregar información en la ventana
    info_img = np.zeros((150, 400, 3), np.uint8)
    cv2.putText(info_img, "Ajuste de Umbrales", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(info_img, "Umbral Inferior: Detecta bordes debiles", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(info_img, "Umbral Superior: Detecta bordes fuertes", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(info_img, "Tip: Inferior < Superior", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(info_img, "Presiona 'r' para resetear", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imshow(window_name, info_img)

def valTrackbars():
    """
    Obtener valores actuales de los trackbars
    """
    window_name = "Controles de Umbralización"
    
    try:
        threshold1 = cv2.getTrackbarPos("Umbral Inferior", window_name)
        threshold2 = cv2.getTrackbarPos("Umbral Superior", window_name)
        
        # Asegurar que threshold1 < threshold2
        if threshold1 >= threshold2:
            threshold2 = threshold1 + 1
            cv2.setTrackbarPos("Umbral Superior", window_name, threshold2)
        
        return threshold1, threshold2
    
    except:
        # Valores por defecto si hay error
        return 200, 200

def enhance_image_quality(img):
    """
    Mejorar calidad de imagen antes del procesamiento
    """
    # Aplicar filtro bilateral para reducir ruido manteniendo bordes
    img_filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Mejorar contraste usando CLAHE
    lab = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    lab = cv2.merge(lab_planes)
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img_enhanced

def auto_adjust_thresholds(img_gray):
    """
    Ajustar automáticamente los umbrales basándose en la imagen
    """
    # Calcular umbral usando método de Otsu
    ret, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calcular umbrales para Canny basándose en Otsu
    lower = 0.5 * ret
    upper = ret
    
    return int(lower), int(upper)

def get_document_corners(contour):
    """
    Obtener las esquinas del documento del contorno
    """
    # Simplificar contorno
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    else:
        # Si no es un cuadrilátero, usar el rectángulo mínimo
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)
        return np.int32(corners)

def calculate_document_area_ratio(contour, img_shape):
    """
    Calcular la relación del área del documento con respecto a la imagen
    """
    contour_area = cv2.contourArea(contour)
    img_area = img_shape[0] * img_shape[1]
    return contour_area / img_area

def is_valid_document_shape(contour, min_area_ratio=0.1, max_area_ratio=0.9):
    """
    Verificar si el contorno tiene una forma válida para un documento
    """
    # Verificar área mínima
    area = cv2.contourArea(contour)
    if area < 5000:
        return False
    
    # Verificar que sea aproximadamente rectangular
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 4:
        return False
    
    # Verificar relación de aspecto
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    aspect_ratio = max(width, height) / min(width, height)
    
    # Los documentos típicos tienen relaciones de aspecto entre 1:1 y 2:1
    if aspect_ratio > 3:
        return False
    
    return True