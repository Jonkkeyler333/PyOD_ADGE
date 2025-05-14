# PyOD_ADGE

Challenge de reproducibilidad, Análisis de datos a gran escala, grupo E

---

## Tutorial de Instalación Local de PyOD_ADGE

### Prerrequisitos

- Python 3.6 o superior instalado.
- Git instalado en tu sistema.
- **Entorno virtual recomendado** (venv, conda, etc.).

---

### Pasos para Instalación

1. **Clonar el Repositorio**

    Abre una terminal y ejecuta:

    ```bash
    git clone https://github.com/Jonkkeyler333/PyOD_ADGE.git
    cd PyOD_ADGE
    ```

2. **Crear un Entorno Virtual** (Opcional pero recomendado)

    - **Con venv (Windows/macOS/Linux):**

      ```bash
      python -m venv myenv
      # Linux/macOS
      source myenv/bin/activate
      # Windows
      myenv\Scripts\activate
      ```

    - **Con Conda (si usas Anaconda):**

      ```bash
      conda create -n pyod_adge_env python=3.8
      conda activate pyod_adge_env
      ```

3. **Instalar Dependencias**

    Instala las dependencias desde `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

4. **Instalar el Paquete en Modo Desarrollo**

    Desde el directorio raíz del proyecto (`PyOD_ADGE/`):

    ```bash
    pip install -e .
    ```

    Esto instalará el paquete en modo editable (los cambios en el código se reflejarán sin reinstalar).

---

### Verificar la Instalación

1. Abre un intérprete de Python:

    ```bash
    python
    ```

2. Intenta importar un módulo del paquete:

    ```python
    from PyOD_ADGE.models import LofParalelizable
    print("¡Paquete instalado correctamente!")
    ```

    Si no hay errores, ¡todo está listo!

---

### Ejemplo de Uso Básico

```python
from PyOD_ADGE.models import LofParalelizable

# Crea una instancia del detector LOF paralelizable
detector = LofParalelizable(n_neighbors=20, contamination=0.1)

# Entrena con tus datos (ejemplo con datos dummy)
import numpy as np
X = np.random.randn(100, 5)  # 100 muestras, 5 características

detector.fit(X)

# Predice anomalías
labels = detector.predict(X)
print(labels)
```

---

### Solución de Problemas Comunes

- **Error de dependencias faltantes:**
  - Asegúrate de ejecutar `pip install -r requirements.txt` antes de `pip install -e .`.

- **Permisos denegados (Linux/macOS):**
  - Usa `sudo` si no estás en un entorno virtual, pero se recomienda usar siempre un entorno virtual.

- **Paquete no reconocido:**
  - Verifica que estás en el directorio correcto (`PyOD_ADGE/`) al ejecutar `pip install -e .`.

---
