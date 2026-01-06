# portfolio\_manager / Robo-UOC-Advisor

This is the code of an app created for a university degree

## Author:

Alberto Garulo Rodríguez (agarulor@uoc.edu)

## Project description

The project is a prototype of a robo-advisor.

The app has been developed in python and Streamlit includes, among others, the following:

* **potfolio.py** --> This file is, as the name suggests, the main program, and it is the one that needs to executed in order to use the app. You can run this file by typing (if you have python installed in your computer) **streamlit run portfolio.py**.
* **src** --> Folder including all the required tools and code to run the app
* **requirements.txt** --> Finally, all the libraries required for the web scraper to work are included in this file. Ideally this is the first file to be run (try with python -m pip install -r requirements.txt)
* **streamlit** --> folder with some basic settings for streamlit
  There is a folder, data, with a .csv file including the initial assets, based on IBEX-35 and EURO STOXX (**data/input/ibex\_eurostoxx.csv**).

License for the APP is Creative Commons Zero v1.0 Universal and the present README file

## How to run it

See Annex II of the Memoria document



Requisitos del sistema:

•	Sistema operativo

o	Windows 10 o superior. El proyecto se realizó sobre Windows 11.

o	Linux. Cualquier distribución más o menos actual.

•	Python.

o	Python 3.11 o superior.



Pasos para la instalación

1\)	Descomprimir archivo ZIP con la aplicación.

2\)	En caso de no disponer de Python. Descargar e instalar Python

a.	En caso de no estar instalado, se puede descargar desde https://www.python.org/

b.	Las versiones actuales permiten marcar la opción de añadir Python al PATH (marcar la opción “ADD Python to PATH”)



Alternativa 1. OPCIÓN MANUAL para instalar librerías, entorno virtual y ejecutar la aplicación 



1\)	(OPCIONAL / RECOMENDADO) Se recomienda el uso de un entorno virtual, para evitar conflictos con otras instalaciones de Python. En caso de querer emplearlas, hacer lo siguiente:

a.	Windows: 

python -m venv .venv  Se ejecuta una sola vez para crearlo.

.venv\\scripts\\activate  Una vez creado solo es necesario ejecutar esta parte (seleccionar opción que considere adecuada, como Z, ejecutar una vez)



b.	Linux:

i.	En algunas distribuciones Linux (especialmente Debian/U.buntu), al intentar crear el entorno virtual puede aparecer el error

“The virtual environment was not created successfully because ensurepip is not available…”

Esto ocurre cuando no está instalado el paquete python3-venv (o el correspondiente a la versión de Python instalada). Para solucionarlo:

1\.	Instalar el paquete necesario:

a.	Para Python 3.11: sudo apt install python3.11-venv

b.	Alternativamente (versión genérica): sudo apt install python3-venv

ii.	Creamos el entorno virtual: python3 -m venv .venv  Se ejecuta una sola vez para crearlo.

iii.	Activarlo: source .venv/bin/activate  Una vez creado solo es necesario ejecutar esta parte.

2\)	(OPCIONAL / RECOMENDADO) Se recomienda actualizar Python. En caso de querer hacerlo:

a.	Windows: 

python -m pip install --upgrade pip

b.	Linux:

pip install --upgrade pip

Puede que la aplicación pida que introduzcas, en el terminal, tu dirección de correo electrónico. No es necesario hacerlo en caso de no querer, bastaría con dar la tecla “entrar” y sería suficiente. El programa se abrirá en el navegador por defecto y a partir de ahí se iniciará la aplicación.

3\)	Una vez descomprimido, en la carpeta donde se ha descomprimido todo, será necesario instalar las librerías necesarias. Estas se encuentran en el archivo “requirements.txt”. Para instalarlas hay que seguir los siguientes pasos (nota, si se ha creado un entorno virtual, asegúrese de que se encuentra activado antes de instalar las librerías).:

a.	Abrir un terminal (Command Prompt en Windows, también puede llamarse Terminal, o Terminal en Linux).

b.	Navegar al directorio raíz de la aplicación (donde se ha descomprimido el archivo ZIP)

c.	Comprobar que el archivo contiene, al menos, las siguientes librerías:

i.	matplotlib==3.10.7

ii.	numpy==2.3.4

iii.	pandas==2.3.3

iv.	plotly==6.5.0

v.	plotly-express==0.4.1

vi.	scipy==1.16.3

vii.	streamlit==1.51.0

viii.	yfinance==0.2.66

d.	Ejecutar:

i.	Windows: python -m pip install -r requirements.txt

ii.	Linux: pip install -r requirements.txt

4\)	Ejecución de la aplicación:

•	Windows: 

o	Ejecutar: streamlit run portfolio.py

•	Linux:

o	Ejecutar: streamlit run portfolio.py



Alternativa 2. OPCIÓN AUTOMÁTICA (crea un entorno virtual, crea las librerías y ejecuta la aplicación) 

En ambos casos, será necesario haber descomprimido el archivo zip y tener disponible el paquete necesario para correr entornos virtuales.

1\)	Windows:

a.	Ejecutar el archivo run\_portfolio.bat, haciendo doble clic sobre él o desde la línea de comandos:

run\_portfolio.bat

2\)	Linux:

a.	Comprobar que el archivo run\_portfolio.sh, tenga los permisos de ejecución. En caso de no tenerlos, dárselos:

chmod 777 run\_portfolio.sh

b.	Ejecutar el archivo desde la línea de comandos:

./run\_portfolio.sh



Puede que la aplicación pida que introduzcas, en el terminal, tu dirección de correo electrónico. No es necesario hacerlo en caso de no querer, bastaría con dar la tecla “entrar” y sería suficiente. El programa se abrirá en el navegador por defecto y a partir de ahí se iniciará la aplicación.



# Disclaimer

Disclaimer

This robo-advisor has been developed exclusively for academic and research
purposes as part of a university final degree project (TFG).

The information, analysis, and outputs generated by this system do not
constitute financial, investment, legal, or tax advice, nor should they be
interpreted as such. The robo-advisor is provided for educational and
experimental purposes only.

Any investment decisions made based on the information produced by this
system are the sole responsibility of the user. The author assumes no
liability for any financial losses, damages, or decisions arising from the
use or misuse of this software.

Past performance, simulations, or hypothetical results are not indicative
of future performance. Financial markets involve risk, and investment
outcomes are inherently uncertain.

By using this software, the user acknowledges and agrees to these terms.

## License

Creative Commons Zero v1.0 Universal

## Postface

I hope you like this work! Feel free to use it as much as you want!

Regards

Alberto Garulo

