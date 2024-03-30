# **Cora Cleaner**
Autoencoder para limpiar ruido cardiaco

![](imgs/result.png?raw=true "Cora Cleaner") 
<br>

## Índice
- [Descripción](#descripción)
- [Repositorio](#repositorio)
- [Instalación](#instalación)
- [Datos](#datos)
    - [Fuentes de Datos](#fuentes-de-datos)
    - [Generación de Datos Sucios](#generación-de-datos-sucios)
    - [Preparación de los Datos](#preparación-de-los-datos)
- [Modelo y Entrenamiento](#modelo-y-entrenamiento)
    - [Arquitectura del Modelo](#arquitectura-del-modelo)
    - [Proceso de Entrenamiento](#proceso-de-entrenamiento)
- [Resultados](#resultados)
    - [Métricas de Desempeño](#métricas-de-desempeño)
    - [Comparación Visual: Audio Original vs. Reconstruido](#comparación-visual-audio-original-vs-reconstruido)
    - [Evaluación Cualitativa](#evaluación-cualitativa)
    - [Conclusiones](#conclusiones)
- [Contribuir](#contribuir)
    - [Reportar Errores](#reportar-errores)
    - [Solicitar Características](#solicitar-características)
    - [Enviar Pull Requests](#enviar-pull-requests)
- [Licencia](#licencia)


--------------------------------------------------------------------------------------------
<br>

## Descripción

Cora Cleaner es un innovador proyecto de aprendizaje profundo diseñado para purificar audios cardiacos, haciéndolos más claros y útiles para diagnósticos precisos. Utilizando datos abiertos de audios cardiacos limpios del Physionet Challenge 2016 y ruidos ambientales hospitalarios, el modelo aprende a eliminar eficazmente el ruido, mejorando significativamente la calidad del sonido. Este enfoque no solo optimiza el análisis de señales cardiacas en entornos ruidosos sino que también abre nuevas posibilidades para la telemedicina y diagnósticos remotos en cardiología.

## Repositorio
    .
    │
    ├── scripts
    │   ├── extract.py        # Función para que a partir de un directorio de audios limpios y con ruido, devolver la lista de los nombres de archivos pares uno a uno.
    │   ├── noiser.py         # Funciones para agregar ruido al audio cardiaco y guardar el nuevo audio "contaminado" rastreable.
    │   ├── plot.py           # Funciones para graficar el audio cardiaco con su spectrograma + su MFCC y para graficar la comparación del audio limpio, sucio y el reconstruido por el modelo.
    │   └── transform.py      # Funciones para generar el espectrograma y el MFC en un Tensor (se experimentó agregarlo como input pero el modelo no mejoró).
    │
    ├── .flake8               # Configuración para mantener buenas prácticas en la legibilidad de código.
    ├── .gitattributes        # Configuración git por default para detectar texto.
    ├── .gitignore            # Configuración para no subir al repositorio distintos tipos de archivo y mantener seguridad de los datos y el modelo.
    ├── LICENSE               # Licencia MIT para mantener buenas prácticas en la protección de propiedad intelectual.
    ├── Pipfile               # Librerías y versionado de las mismas utilizado en este repositorio.
    ├── Pipfile.lock          # "Candado" del Pipfile para utilizar las mismas librerías cuando se ejecute el comando 'pipenv install'.
    │
    └── train.ipynb           # Notebook para visibilizar la configuración del modelo y el entrenamiento del mismo.
<br>


## Instalación

Para comenzar a trabajar con *Cora Cleaner*, sigue estos pasos para configurar tu entorno local. Este proyecto utiliza `pipenv` para manejar las dependencias, asegurando que todos los colaboradores trabajen con las mismas versiones de las librerías.

1. **Clonar el Repositorio**

Primero, clona el repositorio a tu máquina local utilizando Git:

```bash
git clone https://github.com/tu-usuario/cora_cleaner.git
cd cora_cleaner
```

2. **Configurar el Entorno Virtual**

Para crear un entorno virtual y instalar las dependencias necesarias, asegúrate de tener `pipenv` instalado. Si no lo tienes, puedes instalarlo con pip:

```bash
pip install pipenv
```

Una vez instalado `pipenv`, ejecuta el siguiente comando dentro del directorio del proyecto para configurar el entorno virtual y las dependencias:

```bash
pipenv install
```

3. **Activar el Entorno Virtual**

Para activar el entorno virtual y comenzar a trabajar en el proyecto, utiliza:

```bash
pipenv shell
```

Esto te colocará dentro del entorno virtual donde todas las dependencias del proyecto están disponibles.
<br>
<br>
<br>


## Datos

### Fuentes de Datos

Este proyecto utiliza audios cardiacos limpios del *Physionet Challenge 2016* y ruidos ambientales hospitalarios de fuentes abiertas. Estos conjuntos de datos oficiales han sido seleccionados por su calidad y aplicabilidad en simulaciones de entornos reales para el diagnóstico cardiaco.

![](imgs/audio-sample.png?raw=true "Clean audio sample") 


### Generación de Datos Sucios

Para simular condiciones de grabación no ideales, se superponen ruidos hospitalarios a los audios cardiacos limpios. Este proceso se realiza mediante un script que mezcla ruidos a diferentes niveles de intensidad sobre el audio original:

```python
# Ejemplo de cómo se superpone ruido a un audio cardiaco
noised_audio = noiser(audio_path="path/to/clean/audio.wav",
                      noise_path="path/to/noise.wav",
                      noise_louder=-20)  # Intensidad del ruido en dB
```

### Preparación de los Datos

Los datos se preparan en pares: un archivo de audio limpio y su correspondiente versión "sucia". Usamos una función para cargar estos pares, asegurando una asociación correcta para el entrenamiento:

```python
# Cargar pares de audios limpios y con ruido
audio_pairs = load_heart_noised_paths(clean_dir="data/heart_sound_healthy",
                                      noised_dir="data/heart_noised_healthy")
```

Estos fragmentos de código ofrecen una mirada práctica al proceso detrás de la preparación de los datos, desde la generación de audios con ruido hasta la carga de pares para el entrenamiento, facilitando a los interesados en el proyecto una comprensión más profunda de su implementación.
<br>
<br>
<br>



## Modelo y Entrenamiento

### Arquitectura del Modelo

El corazón de *Cora Cleaner* es una red neuronal profunda basada en LSTM, diseñada para abordar eficazmente el desafío de limpiar audios cardiacos. La arquitectura se compone de dos capas LSTM seguidas por tres capas densas, configuradas específicamente para procesar secuencias temporales de audio y extraer características relevantes para la reconstrucción del audio limpio:

- **Capas LSTM:** 2048 unidades cada una, aprovechando su capacidad para retener información a largo plazo, lo que es crucial para entender las dinámicas complejas del audio cardiaco.
- **Capas Densas:** Configuradas con 2048, 1024, y 1024 unidades respectivamente, para refinar y transformar las características extraídas en una señal de audio limpia.

**Hiperparámetros:**
- **Tamaño del Lote:** 50, equilibrando eficiencia computacional y capacidad de generalización.
- **Tasa de Aprendizaje:** Inicialmente 0.001, ajustada dinámicamente con ReduceLROnPlateau para mejorar la convergencia.

![](imgs/model.png?raw=true "Cora Cleaner Model") 

### Proceso de Entrenamiento

El modelo fue entrenado utilizando una división de datos de 70% para entrenamiento, 15% para validación y 15% para pruebas. Este enfoque asegura una evaluación justa de la capacidad del modelo para generalizar a nuevos datos. Se emplearon técnicas como la validación cruzada para garantizar la robustez y la precisión del modelo.

**Validación y Prueba:**
- Se implementó un procedimiento de validación riguroso que monitorea la pérdida de validación para ajustar la tasa de aprendizaje y prevenir el sobreajuste.
- La evaluación final se realizó en el conjunto de prueba, utilizando la pérdida cuadrática media (MSE) como métrica principal para cuantificar la precisión del modelo en la reconstrucción del audio limpio frente al audio original y contaminado.

<br>
<br>
<br>

## Resultados

El modelo *Cora Cleaner* ha demostrado una eficacia impresionante en la tarea de limpiar audios cardiacos contaminados con ruido ambiental. A través de un meticuloso proceso de entrenamiento y validación, hemos logrado afinar el modelo para que ofrezca resultados prometedores.

### Métricas de Desempeño

- **Pérdida Cuadrática Media (MSE) en el Conjunto de Prueba:** 0.0037. Esta métrica indica la precisión del modelo en la reconstrucción del audio limpio, subrayando su capacidad para filtrar eficazmente el ruido del audio cardiaco.

![](imgs/train.png?raw=true "Model training") 

### Comparación Visual: Audio Original vs. Reconstruido

Una de las maneras más claras de apreciar la efectividad de nuestro modelo es mediante la comparación directa entre el audio original (limpio), el audio con ruido (sucio) y el audio reconstruido por el modelo. Esta comparación no solo demuestra la capacidad del modelo para eliminar el ruido, sino también para preservar las características esenciales del audio cardiaco.

```python
# Generación de comparaciones visuales de audio
generate_audio_comparison(original_audio, noisy_audio, reconstructed_audio)
```

![](imgs/result.png?raw=true "Cora Cleaner") 

### Evaluación Cualitativa

Además de las métricas cuantitativas, también realizamos evaluaciones cualitativas del audio reconstruido. Estas evaluaciones involucraron la escucha de audios por parte de expertos, quienes confirmaron la alta calidad del sonido procesado, notando una reducción significativa del ruido con una mínima pérdida de detalles importantes del sonido cardiaco.

### Conclusiones

Los resultados obtenidos subrayan la potencia y la precisión de *Cora Cleaner* en la limpieza de audios cardiacos. Con un MSE Loss notablemente bajo y una fuerte validación cualitativa, el modelo se posiciona como una herramienta prometedora para mejorar el diagnóstico cardiaco remoto y la telemedicina.
<br>
<br>
<br>


## Contribuir

¡Tu contribución es bienvenida y valiosa para el proyecto *Cora Cleaner*! Si estás interesado en mejorar este proyecto, aquí hay varias formas en las que puedes contribuir:

### Reportar Errores

Si encuentras un error, por favor, abre un issue en GitHub describiendo el problema, cómo reproducirlo, y cualquier otro detalle que consideres relevante. Esto nos ayuda a identificar y corregir los errores de manera más eficiente.

### Solicitar Características

¿Tienes ideas sobre nuevas características que podrían mejorar el proyecto? ¡Nos encantaría escucharlas! Abre un nuevo issue, selecciona la opción de solicitud de características y describe tu propuesta con el mayor detalle posible.

### Enviar Pull Requests

Si deseas contribuir directamente al código:

1. **Fork el Repositorio:** Crea tu propio fork del proyecto.
2. **Crea una Nueva Rama:** Para cada nueva característica o corrección, crea una rama específica basada en la rama `main`.
3. **Haz tus Cambios:** Implementa tus cambios, asegurándote de seguir las convenciones de código y documentación del proyecto.
4. **Envía un Pull Request (PR):** Una vez completados tus cambios, envía un PR al repositorio original. En la descripción del PR, explica los cambios realizados, por qué son necesarios, y cualquier otra información que facilite su revisión.

### Guía de Estilo y Convenciones

Te pedimos seguir nuestra guía de estilo y convenciones de código para mantener el proyecto organizado y legible. Puedes encontrar las directrices en el archivo `.flake8` para Python y recomendamos revisar la documentación para familiarizarte con las prácticas recomendadas.

### Mantenerse Actualizado

Para contribuir efectivamente, asegúrate de mantener tu fork y ramas actualizadas con la rama `main` del repositorio original para evitar conflictos durante el merge.

---

Esperamos que estas instrucciones hagan que contribuir al proyecto *Cora Cleaner* sea una experiencia fácil y gratificante. ¡Estamos emocionados por ver tus aportes y trabajar juntos para mejorar esta herramienta!
<br>
<br>
<br>



## Licencia

Este proyecto está bajo la Licencia MIT, lo que permite una amplia libertad para uso personal y comercial, modificación, distribución y uso en proyectos privados, siempre y cuando se incluya el texto original de la licencia y se reconozca el copyright.

### Resumen de la Licencia MIT

La Licencia MIT es una licencia de software libre que pone pocas restricciones en el uso del software. Bajo esta licencia, se permite:

- **Uso Comercial:** Puedes utilizar el software para proyectos comerciales sin restricciones.
- **Modificación:** Tienes la libertad de modificar el software según tus necesidades.
- **Distribución:** Puedes compartir el software con quien quieras, bajo los términos de esta licencia.
- **Uso en Proyectos Privados:** Puedes utilizar el software en proyectos privados sin preocupaciones sobre derechos de autor.

Para más detalles sobre la Licencia MIT, por favor revisa el archivo `LICENSE` incluido en este repositorio.
