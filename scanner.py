import cv2
import numpy as np
import utils
import sys

print("Iniciando Document Scanner...")

########################################################################


# Preguntar si se quiere usar la webcam o una imagen
webCamFeed = input("¿Usar webcam? (s/n): ").strip().lower() == 's'
default_image = "1.jpg"
default_url = "http://10.7.121.249:8080/video"

# Si no se usa la webcam, se solicita la ruta de un archivo de Imagen
if not webCamFeed:
    pathImage = input("Ingrese la ruta del archivo de Imagen: ").strip()
    if not pathImage:
        print("Error: No se proporcionó una ruta de archivo de Imagen.")
        sys.exit(1)
else:
    # Si se usa la webcam, se solicita la URL de la cámara IP (si es necesario)
    url = input("Ingrese la URL de la cámara IP (por ejemplo, 0 para cámara local): ").strip()
    if not url:
        print("Error: No se proporcionó una URL de cámara IP.")
        url = default_url
    print("Usando cámara IP:", url)
    cap = cv2.VideoCapture(url)  # Se usa la cámara IP en lugar de la webcam local

# Verificar si la cámara está disponible
if webCamFeed == True and not cap.isOpened():
    print("Error: No se pudo acceder a la cámara")
    print("Cambiando a modo cámara local...")
    cap = cv2.VideoCapture(0) 
if webCamFeed == True and not cap.isOpened():
    print("Error: No se pudo acceder a la cámara")
    print("Cambiando a modo imagen estática...")
    webCamFeed = False

if webCamFeed == True: 
    cap.set(10,160)
heightImg = 640
widthImg  = 480
########################################################################
 
utils.initializeTrackbars()
count=0
 
while True:
 
    if webCamFeed:
        success, img = cap.read()
        if not success:
            print("Error: No se pudo capturar frame de la cámara")
            break
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print(f"Error: No se pudo cargar la imagen {pathImage}")
            break
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE    
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=utils.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
 
    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
     # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utils.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest=utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utils.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
 
        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
 
        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)        # Image Array for Display
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])
 
    else:
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
 
    # LABELS FOR DISPLAY
    labels = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]
 
    stackedImage = utils.stackImages(imageArray,0.75,labels)
    cv2.imshow("Result",stackedImage)
 
    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
    # Salir del bucle si se presiona la tecla 'q'
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        print("Saliendo del escáner de documentos...")
        break
# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

