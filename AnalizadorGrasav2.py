
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

        if self.conn is None or not self.conn.is_connected():

            self.initDB()

        data_group_id = self.getDataGroupId()

        logging.debug(
            f"getModelData: idAnalizador={self.idAnalizador}"
        )

        logging.debug(
            f"getModelData: dataGroupId={data_group_id}"
        )

        if data_group_id <= 0:

            return False

        try:

            query = """
                SELECT
                    temperatura,
                    valorInstrumento,
                    valorLaboratorio
                FROM muestrasGrasa
                WHERE seleccionada = 1
                AND valorLaboratorio > 0
                AND idVersionMuestra = %s
                AND idAnalizador = %s
            """

            self.cursor.execute(
                query,
                (
                    data_group_id,
                    self.idAnalizador
                )
            )

            data = self.cursor.fetchall()

            self.X = []
            self.Y = []

            for row in data:

                self.X.append(
                    [
                        row["temperatura"],
                        row["valorInstrumento"]
                    ]
                )

                self.Y.append(
                    [
                        row["valorLaboratorio"]
                    ]
                )

            logging.info(
                f"Recuperadas {len(data)} muestras"
            )

            return len(data) > 0

        except mysql.connector.Error:

            logging.exception(
                "Error obteniendo datos del modelo"
            )

            return False