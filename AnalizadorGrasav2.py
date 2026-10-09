
# Version 1.0 22/06/2026 
# Coding: UTF-8
# Fernando Soto


#Librerias
import sys,time,os # Manejo de pthon, sistema operativo y tiempos.
import mysql.connector # Conector a la base de datos.
from asyncua import Client, ua # Cliente OPC
import threading # Ejecutar varias tareas de manera simultanea
import logging # Permite registrar eventos, errores y mensajes
from dotenv import load_dotenv


# cargar archivo de configuración
load_dotenv()

#El logging se almacena en un fichero llamado: "AnalizadorGrasa.log", y se genera cada vez que arranca el programa: filemode="w". EL nivel debug minimo es DEBUG
logging.basicConfig(
        filename=os.getenv("LOG_PATH"), 
        filemode="a", 
        level=logging.DEBUG 
)

#Libreria Machine Learning #pip 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import joblib

# Libreria para correr tareas periodicas en segundo plano
from apscheduler.schedulers.background import BackgroundScheduler

# Configuración de planta

urlServidorOPC = os.getenv("URL_SERVIDOR_OPC")
codPlanta = os.getenv("COD_PLANTA")

tagPathInstancia_1 = os.getenv("TAGPATH_INSTANCIA_1")
tagPathInstancia_4 = os.getenv("TAGPATH_INSTANCIA_4")

idInstancia_1 = int(os.getenv("ID_INSTANCIA_1"))
idInstancia_4 = int(os.getenv("ID_INSTANCIA_4"))

# Frecuencia de muestreo
FREQ_SAMPLE = int(os.getenv("FREQ_SAMPLE"))

# Credenciales
OPC_USER = os.getenv("OPC_USER")
OPC_PASSWORD = os.getenv("OPC_PASSWORD")
URL_SERVIDOR_OPC = urlServidorOPC
OPC_PORT = int(os.getenv("OPC_PORT"))
TAGPATH_WACHDOG = os.getenv("WATCHDOG_PATH")
COUNT_LIMIT = 5
COUNT = 0

# Tiempos en segundos.
TIME_SAMPLE = int(os.getenv("TIME_SAMPLE"))
TIME_FOR_REINIT = 0
TIME_FOR_REINIT_LIMIT = int(os.getenv("TIME_FOR_REINIT_LIMIT"))

# Esta clase representa la conexión al servidor OPC UA.

import asyncio
import logging
from asyncua import Client


class OPCManager:

    def __init__(self, server_ip, port, user, password):

        self.server_ip = server_ip
        self.port = port
        self.user = user
        self.password = password

        self.client = None

    async def connect(self):

        connection_string = (
            f"opc.tcp://{self.user}:{self.password}"
            f"@{self.server_ip}:{self.port}/iaopcua/None"
        )

        self.client = Client(connection_string)

        retries = 0
        max_retries = 30

        while retries < max_retries:

            try:

                await self.client.connect()

                logging.info(
                    "Conectado correctamente al servidor OPC"
                )

                return True

            except Exception:

                retries += 1

                logging.exception(
                    f"Intento de conexión {retries}/{max_retries}"
                )

                await asyncio.sleep(5)

        raise RuntimeError(
            "No fue posible conectar con OPC"
        )

    async def read(self, tag):

        try:

            node = self.client.get_node(tag)

            return await node.read_value()

        except Exception:

            logging.exception(
                f"Error leyendo tag OPC: {tag}"
            )

            return None

    async def write(self, tag, value):

        try:

            node = self.client.get_node(tag)

            await node.write_value(value)

            return True

        except Exception:

            logging.exception(
                f"Error escribiendo tag OPC: {tag}"
            )

            return False

    async def disconnect(self):

        try:

            await self.client.disconnect()

        except Exception:

            logging.exception(
                "Error desconectando OPC"
            )


class WatchdogMonitor:

    def __init__(self, opc, tag, limit):

        self.opc = opc
        self.tag = tag
        self.limit = limit

        self.fail_count = 0

    def check(self):

        value = self.opc.read(self.tag)

        if value is None:

            self.fail_count += 1

        elif value:

            self.opc.write(
                self.tag,
                False
            )

            self.fail_count = 0

        else:

            self.fail_count += 1

        if self.fail_count >= self.limit:

            raise RuntimeError(
                "Watchdog perdido"
            )            
        


class MedicionGrasa:
    
    tagNameIdGrupoMuestras = "idVersionMuestras"
    tagReturnedResult = "valorCorregido"
    tagWatchdog = "watchdog"
    tagCoefR2Min = "coefR2Min"
    tagNumMuestrasMin = "numMuestrasMin"
    tagCoefR2 = "coefR2"
    tagFitModel = "fitModel"
    tagAlarm = "alarmas"
    tagDataIsValid = "dataIsValid"
    tagStop = "stop"
    tagValorCorregido = "valorCorregido"
    tagValorTemperatura = "valorTemperatura"
    tagValorMedicionGrasa = "valorMedicionGrasa"
    defaultProvider = "ns=2;s=[default]"

    alarmas = {1: "El numero de muestras no es suficiente",
               2: "El coeficiente R^2 es inferior al minimo establecido",
               3: "Error de conexion con la base de datos"}

    def __init__(self, urlServidorOPC, pathUDT, idAnalizador, client):
        self.pathUDT = pathUDT
        self.urlServidorOPC = urlServidorOPC
        self.idAnalizador= idAnalizador
        self.tagPathTemp = f"(self.defaultProvider, self.pathUDT, self.tagValorTemperatura)"
        self.tagPathMedidorPS = f"(self.defaultProvider, self.pathUDT, self.tagValorMedicionGrasa)"
        self.tagPathValorCorregido = f"(self.defaultProvider, self.pathUDT, self.tagValorCorregido)"
        self.tagDataIsValid = f"(self.defaultProvider, self.pathUDT, self.tagDataIsValid)"

        self.modelName = "model_%s.pkl" % pathUDT.split("/")[-1]

   
        #Datos de conexion con la base de datos
        self.config = {
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "host": os.getenv("MYSQL_HOST"),
            "database": os.getenv("MYSQL_DATABASE"),
            "raise_on_warnings": True,
            "charset": "utf8"
        }

        #El coeficiente de determinacion R^2 con el que se valido el modelo durante las pruebas fue de 0.999992
        self.coefR2Min = float(
            os.getenv("COEF_R2_MIN")
        )
        #El numero de muestras con el que se valido el modelo fue de 1474
        self.numMuestrasMin = int(
            os.getenv("NUM_MUESTRAS_MIN")
        )
        #Con esta variable paramos el proceso
        self.stop = False

        #Inicializamos la base de datos
        self.initDB()

        self.client = client

        #Array con los datos del modelo
        self.X = [] # Temperatura
        self.Y = [] # Grasa corregida

        #Obtenemos los datos del modelo
        self.getModelData()

        #Si existia un modelo creado previamente lo recuperamos, si no, creamos uno nuevo
        #if not self.recoverModel():
        self.initMachineLearning()
        self.fitMachineLearning(None, 1)

        #Estado del watchdog
        self.watchdogStatus = False

        #Nos subscribimos a los tags de parametros y ordenes para detectar cambios
        self.subs = []
        self.initSubscription()

    #Clase auxiliar que se emplea cuando se produce un datachange de un tag OPC
    class SubHandler:

        def __init__(self, callback):
            self.callback = callback

        def datachange_notification(
            self,
            node,
            val,
            data
        ):

            threading.Thread(
                target=self.callback,
                args=(node, val),
                daemon=True
            ).start()

    async def initSubscription(self):

        """
        Crear todas las suscripciones OPC necesarias
        """

        #
        # Parámetros
        #

        tag_path_coef_r2_min_forz = (
            f"{self.defaultProvider}"
            f"{self.pathUDT}"
            f"/PAR/{self.tagCoefR2Min}Forz"
        )

        tag_path_num_muestras_min_forz = (
            f"{self.defaultProvider}"
            f"{self.pathUDT}"
            f"/PAR/{self.tagNumMuestrasMin}Forz"
        )

        node_coef_r2_min_forz = self.client.get_node(
            tag_path_coef_r2_min_forz
        )

        node_num_muestras_min_forz = self.client.get_node(
            tag_path_num_muestras_min_forz
        )

        self.subParams = await self.client.create_subscription(
            1000,
            self.SubHandler(self.setParam)
        )

        await self.subParams.subscribe_data_change(
            node_coef_r2_min_forz
        )

        await self.subParams.subscribe_data_change(
            node_num_muestras_min_forz
        )

        self.subs.append(self.subParams)

        #
        # Fit Model
        #

        tag_path_fit_model = (
            f"{self.defaultProvider}"
            f"{self.pathUDT}"
            f"/{self.tagFitModel}"
        )

        node_fit_model = self.client.get_node(
            tag_path_fit_model
        )

        self.subFitModel = await self.client.create_subscription(
            2000,
            self.SubHandler(self.fitMachineLearning)
        )

        await self.subFitModel.subscribe_data_change(
            node_fit_model
        )

        self.subs.append(self.subFitModel)

        #
        # Stop
        #

        tag_path_stop = (
            f"{self.defaultProvider}"
            f"{self.pathUDT}"
            f"/{self.tagStop}"
        )

        node_stop = self.client.get_node(
            tag_path_stop
        )

        self.subStop = await self.client.create_subscription(
            2000,
            self.SubHandler(self.stopApplication)
        )

        await self.subStop.subscribe_data_change(
            node_stop
        )

        self.subs.append(self.subStop)

        logging.info(
            f"Suscripciones OPC inicializadas para "
            f"analizador {self.idAnalizador}"
        )

    def recoverModel(self):

        try:

            self.model = joblib.load(
                self.modelName
            )

            logging.info(
                f"Modelo recuperado: {self.modelName}"
            )

            return True

        except Exception:

            logging.exception(
                f"Error recuperando modelo {self.modelName}"
            )

            return False


    def initDB(self):

        try:

            self.conn = mysql.connector.connect(
                **self.config
            )

            self.cursor = self.conn.cursor(
                dictionary=True
            )

            self.conn.autocommit = True

            logging.info(
                "Conexión MySQL establecida"
            )

        except mysql.connector.Error:

            logging.exception(
                "Error conectando con MySQL"
            )

            raise


    def secondCycle(self):

        logging.debug("Ciclo de 1 segundo")

        self.watchdog()

        self.predictValue()

    def closeDB(self):

        try:

            if self.cursor:

                self.cursor.close()

            if self.conn and self.conn.is_connected():

                self.conn.close()

            logging.info(
                "Conexión MySQL cerrada"
            )

        except mysql.connector.Error:

            logging.exception(
                "Error cerrando conexión MySQL"
            )        

    def getModelData(self):
 
        globalResult = 0

        # Inicializar la conexión si no está abierta
        if self.conn is None or not self.conn.is_connected():
            self.initDB()

        # Obtener el ID del grupo de datos
        dataGroupId = self.getDataGroupId()

        logging.debug(f"getModelData; idAnalizador = {self.idAnalizador}")
        logging.debug(f"getModelData; dataGroupId = {dataGroupId}")

        if dataGroupId > 0:
            try:
                # Asegúrate de habilitar dictionary=True en self.initDB() al crear el cursor,
                # o maneja los resultados por índice si usas un cursor estándar.
                query = """
                    SELECT temperatura, valorInstrumento, valorLaboratorio 
                    FROM muestrasGrasa 
                    WHERE seleccionada = 1 
                    AND valorLaboratorio > 0 
                    AND idVersionMuestra = %s 
                    AND idAnalizador = %s
                """
                self.cursor.execute(query, (dataGroupId, self.idAnalizador))
                data = self.cursor.fetchall()

                self.X = []
                self.Y = []

                if data:
                    # Si el cursor es de tipo Diccionario (dictionary=True):
                    if isinstance(data[0], dict):
                        self.X = [
                            [row["temperatura"], row["valorInstrumento"]]
                            for row in data
                        ]
                        self.Y = [[row["valorLaboratorio"]] for row in data]
                    # Si el cursor devuelve Tuplas (cursor por defecto):
                    else:
                        matrix = np.array(data, dtype=float)
                        self.X = matrix[:, :2]  # Columnas: temperatura, valorInstrumento
                        self.Y = matrix[:, 2:]  # Columna: valorLaboratorio

                    globalResult = 1

            except Exception as e:
                # logging.exception adjunta la traza de error completa sin necesidad de sys.exc_info()
                logging.exception(
                    "getModelData: Excepción al obtener los datos del modelo."
                )

        return globalResult

# Obtenemos el grupo de muestras actualmente abierto
    def getDataGroupId(self):

        # Limpiamos barras sobrantes en los bordes para garantizar un path OPC válido
        provider = self.defaultProvider.rstrip("/")
        path = self.pathUDT.strip("/")
        tag_name = self.tagNameIdGrupoMuestras.lstrip("/")

        # Construcción dinámica del path usando f-string
        tagDataGroupId = f"{provider}/{path}/{tag_name}"

        try:
            val = self.getOPCValue(tagDataGroupId)
            return int(val) if val is not None else 0
        except (ValueError, TypeError) as e:
            logging.error(
                f"getDataGroupId: Error al convertir el valor del tag '{tagDataGroupId}'. Error: {e}"
            )
            return 0

    # Inicializamos el servicio OPC
    def initOPC(self):
        """Inicializa y establece la conexión con el servidor OPC UA."""
        # Construcción de la URL endpoint
        opc_url = f"opc.tcp://{OPC_USER}:{OPC_PASSWORD}@{self.urlServidorOPC}:{OPC_PORT}/iaopcua/None"

        logging.info("Iniciando conexión al servidor OPC UA...")
        logging.debug(f"Endpoint OPC UA: {opc_url}")

        try:
            # Creación del cliente OPC
            self.client = Client(opc_url)

            # Realizamos la conexión
            self.client.connect()
            logging.info("Conexión OPC UA establecida con éxito.")

        except Exception as e:
            logging.error(
                f"Error al conectar con el servidor OPC UA en {self.urlServidorOPC}: {e}"
            )

            raise

    # Cerramos el cliente OPC
    def closeOPCClient(self):
    
        # Cerrar suscripciones activas si la lista existe
        if hasattr(self, "subs") and self.subs:
            logging.info(
                f"Eliminando {len(self.subs)} suscripción(es) OPC UA..."
            )
            for sub in self.subs:
                try:
                    sub.delete()
                except Exception as e:
                    logging.warning(
                        f"closeOPCClient: Error al eliminar suscripción: {e}"
                    )
            # Vaciar la lista de suscripciones
            self.subs.clear()

        # Desconectar el cliente OPC si existe y está inicializado
        if hasattr(self, "client") and self.client is not None:
            try:
                logging.info("Desconectando cliente OPC UA...")
                self.client.disconnect()
                logging.info("Cliente OPC UA desconectado correctamente.")
            except Exception as e:
                logging.error(
                    f"closeOPCClient: Error al desconectar el cliente OPC UA: {e}"
                )

    # Inicializamos el modelo de machine learning
    def initMachineLearning(self, degree=3, alpha=1.0):
      
        try:
            # Creamos el pipeline combinando la expansión polinómica y el modelo de regresión Ridge
            self.model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=True),
                Ridge(alpha=alpha, random_state=42),
            )
            logging.info(
                f"initMachineLearning: Modelo Ridge con PolynomialFeatures (grado={degree}, alpha={alpha}) inicializado con éxito."
            )
        except Exception as e:
            logging.exception(
                "initMachineLearning: Error al inicializar el modelo de Machine Learning."
            )
            raise

    def fitMachineLearning(self, node, value):

        logging.debug(f"fitMachineLearning: value={value}")

        # Ejecutamos el modelo si el bit de orden es True/1 y los datos son válidos
        if value and self.getModelData():
            # Aseguramos que Y sea un arreglo 1D si es una sola columna objetivo
            X_arr = np.array(self.X)
            Y_arr = np.array(self.Y).ravel()

            # Verificación mínima de muestras suficientes antes de la validación cruzada
            if len(X_arr) < 2:
                logging.warning(
                    "fitMachineLearning: Muestras insuficientes para entrenar el modelo."
                )
                return

            try:
                # Entrenamiento del modelo
                self.model.fit(X_arr, Y_arr)

                # Configuración del esquema de validación cruzada
                # Si hay pocas muestras, limitamos n_splits para evitar errores
                n_splits = min(10, len(X_arr))
                self.cv = ShuffleSplit(
                    n_splits=n_splits, test_size=0.3, random_state=0
                )

                # Cálculo del R^2 medio mediante validación cruzada
                scores = cross_val_score(self.model, X_arr, Y_arr, cv=self.cv)
                self.cross_val_score = float(np.mean(scores))

                # Guardado del modelo en disco usando joblib moderno
                joblib.dump(self.model, self.modelName)
                logging.debug(
                    f"fitMachineLearning: Resultado de la validación cruzada R^2: {self.cross_val_score:.4f}"
                )

                # Enviar el coeficiente R2 actualizado al SCADA
                self.setCoefR2()

                # Reseteo del bit de orden en el servidor OPC UA
                if node:
                    try:
                        # Se asigna False / 0 para confirmar el procesamiento de la orden
                        node.set_value(False)
                        logging.info(
                            "fitMachineLearning: Bit de orden reseteado a 0 con éxito."
                        )
                    except Exception as e:
                        logging.warning(
                            f"fitMachineLearning: No se pudo resetear el bit de orden en el nodo OPC. Error: {e}"
                        )

            except Exception as e:
                logging.exception(
                    "fitMachineLearning: Excepción durante el proceso de entrenamiento del modelo."
                )
        else:
            logging.warning(
                "fitMachineLearning: Atención, no se genera el modelo (orden inactiva o sin datos de entrenamiento)."
            )

    # Función auxiliar para obtener un valor del servidor OPC a partir de un tag
    def getOPCValue(self, tagPath):

        try:
            node = self.client.get_node(tagPath)
            value = node.get_value()
            logging.debug(f"getOPCValue: tagPath: {tagPath}, value: {value}")
            return value

        except Exception as e:
            logging.error(
                f"getOPCValue: Excepción al leer el tag '{tagPath}' del servidor OPC. Error: {e}"
            )
            return None

    # Función auxiliar para escribir un valor en un tag del servidor OPC
    def setOPCValue(self, tagPath, value):

        try:
            node = self.client.get_node(tagPath)

            # Escribir directamente el valor
            node.set_value(value)

            # logging.debug(f"setOPCValue: tagPath: {tagPath}, value: {value}")
            return True

        except Exception as e:
            logging.error(
                f"setOPCValue: Excepción al escribir el valor '{value}' en el tag '{tagPath}'. Error: {e}"
            )
            return False

    # Lee los valores de temperatura y de grasa y calcula el valor corregido
    def predictValue(self):
    
        # Solo ejecutamos la función si el modelo es válido
        if self.isModelValid():
            # Obtenemos los valores desde el servidor OPC UA
            temperatura = self.getOPCValue(self.tagPathTemp)
            medidaGrasa = self.getOPCValue(self.tagPathMedidorPS)

            logging.debug(
                f"predictValue: Valor de temperatura: {temperatura} C, Valor del medidor de grasa: {medidaGrasa} PS"
            )

            # Marcamos inicialmente el dato como válido
            self.setValidData(1)

            # Verificamos que los valores leídos existan y sean mayores que 0
            if (
                temperatura is not None
                and medidaGrasa is not None
                and temperatura > 0
                and medidaGrasa > 0
            ):
                try:
                    # Predecimos el valor pasando una matriz 2D de entrada
                    prediction = self.model.predict([[temperatura, medidaGrasa]])

                    # Obtenemos el escalar sin importar si prediction es 1D o 2D
                    valorCorregido = float(np.ravel(prediction)[0])

                    # Acotamos el resultado entre 0 y 5
                    valorCorregido = min(max(0.0, valorCorregido), 5.0)

                    logging.debug(
                        f"predictValue: Valor corregido calculado: {valorCorregido:.4f}"
                    )

                    # Escribimos el valor corregido en el tag correspondiente del servidor OPC
                    self.setOPCValue(
                        self.tagPathValorCorregido, valorCorregido
                    )
                    return valorCorregido

                except Exception as e:
                    logging.exception(
                        "predictValue: Excepción al predecir el valor corregido."
                    )
                    self.setValidData(0)
                    return 0
            else:
                logging.warning(
                    "predictValue: Mediciones analógicas fuera de rango o lecturas nulas."
                )
                self.setValidData(0)
                return 0
        else:
            self.setValidData(0)
            return 0

    # Escribimos el bit de datos válidos
    def setValidData(self, isValid):
      
        # Convertimos a valor booleano explícito para evitar incompatibilidad de tipos en OPC/SCADA
        status = bool(isValid)
        self.setOPCValue(self.tagDataIsValid, status)

    # Comprobamos si el modelo es válido
    def isModelValid(self):

        try:
            # Verificar si los atributos existen e inicializados
            if not hasattr(self, "X") or self.X is None:
                logging.debug("isModelValid: self.X no está inicializado.")
                return False

            if (
                not hasattr(self, "cross_val_score")
                or self.cross_val_score is None
            ):
                logging.debug(
                    "isModelValid: self.cross_val_score no está calculado."
                )
                return False

            # Comprobación de condiciones mínimas
            num_muestras_ok = len(self.X) > self.numMuestrasMin
            coef_r2_ok = self.cross_val_score > self.coefR2Min

            return num_muestras_ok and coef_r2_ok

        except Exception as e:
            logging.error(
                f"isModelValid: Excepción al verificar la validez del modelo. Error: {e}"
            )
            return False

    # Guardamos los parametros y confirmamos la recepcion
    def setParam(self, node, value):
        
        try:
            browse_name = node.get_browse_name()
            namespace_idx = browse_name.NamespaceIndex
            node_name = browse_name.Name

            tagPath = f"ns={namespace_idx};s={node_name}"
            # logging.debug(f"setParam: Estableciendo el parámetro con tag: {tagPath} y valor: {value}")

            # Obtenemos el nombre del atributo a partir del nombre del path
            nomAtributo = node_name.split("/")[-1].replace("Forz", "")
            setattr(self, nomAtributo, value)

            # Escribimos el valor en el tag correspondiente de visualización (Visu)
            tagPathVisu = tagPath.replace("Forz", "Visu")
            self.setOPCValue(tagPathVisu, value)

        except Exception as e:
            logging.error(f"setParam: Excepción al establecer el parámetro. Error: {e}")

    # Establecemos el número de la alarma
    def setAlarm(self, alarmNumber):
        
        try:
            # Construcción limpia del path del tag de la alarma
            provider = self.defaultProvider.rstrip("/")
            path = self.pathUDT.strip("/")
            tag = self.tagAlarm.lstrip("/")
            tagPathAlarm = f"{provider}/{path}/{tag}"

            # Obtenemos la alarma actual (si falla o devuelve None, asumimos 0)
            currentAlarm = self.getOPCValue(tagPathAlarm)
            if currentAlarm is None or not isinstance(currentAlarm, int):
                currentAlarm = 0

            # Seteamos el bit de la alarma que se ha generado mediante máscara binaria
            newAlarmValue = (1 << alarmNumber) | currentAlarm
            self.setOPCValue(tagPathAlarm, newAlarmValue)

        except Exception as e:
            logging.error(f"setAlarm: Excepción al establecer la alarma {alarmNumber}. Error: {e}")

    # Limpiamos todas las alarmas
    def clearAlarms(self):
        
        try:
            provider = self.defaultProvider.rstrip("/")
            path = self.pathUDT.strip("/")
            tag = self.tagAlarm.lstrip("/")
            tagPathAlarm = f"{provider}/{path}/{tag}"

            self.setOPCValue(tagPathAlarm, 0)

        except Exception as e:
            logging.error(f"clearAlarms: Excepción al limpiar las alarmas. Error: {e}")

    # Guardamos los parámetros y confirmamos la recepción para detener la aplicación
    def stopApplication(self, node, value):
        
        logging.info(f"stopApplication: Orden de parada recibida (value={value}).")

        # Se activa la orden de parada si el valor es True o 1
        if value in (1, True):
            # Resetear el bit de orden en el nodo OPC como confirmación
            if node:
                try:
                    node.set_value(False)
                    logging.info(
                        "stopApplication: Bit de orden de parada reseteado a 0/False con éxito."
                    )
                except Exception as e:
                    logging.warning(
                        f"stopApplication: No se ha podido resetear el bit de orden. Error: {e}"
                    )

            # Establecer la bandera de parada para detener los bucles/scheduler
            self.stop = True

    # Escribimos el coeficiente R2 calculado
    def setCoefR2(self):
        
        try:
            # Construcción limpia del path del tag R2
            provider = self.defaultProvider.rstrip("/")
            path = self.pathUDT.strip("/")
            tag = self.tagCoefR2.lstrip("/")
            tagPathCoefR2 = f"{provider}/{path}/{tag}"

            # Verificamos si existe el atributo cross_val_score, en caso contrario enviamos 0.0
            r2_value = getattr(self, "cross_val_score", 0.0)
            if r2_value is None:
                r2_value = 0.0
            else:
                r2_value = float(r2_value)

            logging.debug(
                f"setCoefR2: Escribiendo R^2 ({r2_value:.4f}) en tag: {tagPathCoefR2}"
            )
            self.setOPCValue(tagPathCoefR2, r2_value)

        except Exception as e:
            logging.error(
                f"setCoefR2: Excepción al escribir el coeficiente R^2. Error: {e}"
            )

    # Cambia el valor del bit de watchdog
    def watchdog(self):
        
        try:
            # Construcción limpia del path del tag watchdog
            provider = self.defaultProvider.rstrip("/")
            path = self.pathUDT.strip("/")
            tag = self.tagWatchdog.lstrip("/")
            tagWatchdog = f"{provider}/{path}/{tag}"

            # Inicializamos o conmutamos la variable del estado del watchdog
            if not hasattr(self, "watchdogStatus"):
                self.watchdogStatus = False

            self.watchdogStatus = not self.watchdogStatus

            # Enviamos el nuevo valor booleano
            self.setOPCValue(tagWatchdog, self.watchdogStatus)

        except Exception as e:
            logging.error(
                f"watchdog: Excepción al conmutar el bit de watchdog. Error: {e}"
            )

    # Esta función arranca las tareas de la aplicación
    def run(self):        
        try:
            logging.info("Iniciando servicio y planificador de tareas...")

            # Creamos el scheduler que ejecuta secondCycle periódicamente
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(
                func=self.secondCycle,
                trigger="cron",
                second=f"*/{int(FREQ_SAMPLE)}",
                max_instances=1,
                misfire_grace_time=3,
            )
            self.scheduler.start()
            logging.info(
                f"Scheduler iniciado correctamente con frecuencia de {FREQ_SAMPLE} s."
            )

            # Creamos el hilo que monitorea la orden de parada
            t = threading.Thread(target=self.end, daemon=True)
            t.start()

        except Exception as e:
            logging.exception("run: Excepción al arrancar la aplicación.")

    # Esta función corre en un hilo separado y se encarga de parar la aplicación cuando la orden stop se activa
    def end(self):
        
        while not getattr(self, "stop", False):
            time.sleep(0.1)

        logging.info("stopApplication: Parando la aplicación...")
        self.closeAll()

    # Cierra todas las conexiones de los servicios y procesos abiertos
    def closeAll(self):
        
        logging.info("Iniciando cierre de todos los servicios...")

        # Detener el Scheduler
        if hasattr(self, "scheduler") and self.scheduler.running:
            try:
                self.scheduler.shutdown(wait=False)
                logging.info("Scheduler detenido con éxito.")
            except Exception as e:
                logging.error(f"closeAll: Error al detener el scheduler: {e}")

        # Cerrar cliente OPC UA
        try:
            self.closeOPCClient()
        except Exception as e:
            logging.error(
                f"closeAll: Error al cerrar el cliente OPC UA: {e}"
            )

        # Cerrar conexión con la base de datos
        try:
            if hasattr(self, "closeDB"):
                self.closeDB()
                logging.info("Conexión a la base de datos cerrada con éxito.")
        except Exception as e:
            logging.error(
                f"closeAll: Error al cerrar la base de datos: {e}"
            )

        logging.info("Todos los servicios han sido cerrados correctamente.")

if __name__ == "__main__":
    try:
        # Inicialización del cliente OPC global
        client = initOPC()
        logging.info(
            "------------> CONEXIÓN ESTABLECIDA CON SERVIDOR OPC UA"
        )

        # Configuración del watchdog vía suscripción OPC UA
        wachdog_node = client.get_node(TAGPATH_WACHDOG)
        wachdog_sub = client.create_subscription(
            200, SubHandler(reinit_count)
        )
        wachdog_sub.subscribe_data_change(wachdog_node)

        # Instancias de la clase MedicionGrasa
        estandarizador1 = MedicionGrasa(
            urlServidorOPC, tagPathInstancia_1, idInstancia_1, client
        )
        estandarizador4 = MedicionGrasa(
            urlServidorOPC, tagPathInstancia_4, idInstancia_4, client
        )

        # Arranque de los servicios periódicos
        estandarizador1.run()
        estandarizador4.run()

        logging.info("El hilo principal se encuentra en ejecución...")

        # Bucle principal de vigilancia / monitorización
        while True:
            time.sleep(TIME_SAMPLE)
            TIME_FOR_REINIT += TIME_SAMPLE

            # Comprobación de tiempo límite para reinicio por Watchdog
            if TIME_FOR_REINIT > TIME_FOR_REINIT_LIMIT:
                logging.warning(
                    f"Tiempo límite de reinit superado ({TIME_FOR_REINIT}s). Iniciando parada segura y reinicio..."
                )

                # Cierre ordenado de los servicios e instancias
                try:
                    estandarizador1.closeAll()
                    estandarizador4.closeAll()
                    wachdog_sub.delete()
                    client.disconnect()
                except Exception as e:
                    logging.error(f"Error durante el cierre ordenado: {e}")

                # Reinicio del proceso / Terminación limpia
                # Si utilizas un gestor de procesos (NSSM, Systemd, Docker, etc.), basta con terminar con código 1 o 0:
                logging.info("Terminando proceso...")

                # Mantenemos la opción de terminación por sistema si es estrictamente necesario:
                if os.name == "nt":
                    os.system(f"Taskkill /PID {os.getpid()} /F")
                else:
                    os.system(f"kill -9 {os.getpid()}")

                sys.exit(0)

    except KeyboardInterrupt:
        logging.info(
            "Interrupción por teclado detectada (Ctrl+C). Parando aplicación..."
        )
        try:
            estandarizador1.closeAll()
            estandarizador4.closeAll()
            if "client" in locals():
                client.disconnect()
        except Exception as e:
            logging.error(f"Error al cerrar recursos en interrupción: {e}")

        sys.exit(0)

    except Exception as e:
        logging.exception(
            "Excepción no controlada en el hilo principal del programa."
        )
        sys.exit(1)