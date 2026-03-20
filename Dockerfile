# Imagen base con Python 3.8 sobre Ubuntu
FROM python:3.8-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo dentro del contenedor
WORKDIR /workspace

# Copiar el repo al contenedor
COPY . .

# Downgrade setuptools PRIMERO para que gym antiguo pueda instalarse
RUN pip install "setuptools==65.5.0"

# Core dependencies con versiones corregidas
RUN pip install coloredlogs==15.0.1
RUN pip install numpy==1.22.0
RUN pip install pandas==1.2.4
RUN pip install pomegranate==0.14.6
RUN pip install psutil==5.8.0
RUN pip install scipy==1.10.0
RUN pip install termcolor
RUN pip install "ray[rllib]==1.13.0"

# Dev dependencies
RUN pip install p_tqdm==1.3.3
RUN pip install pre-commit==2.13
RUN pip install pytest==6.2.4
RUN pip install pytest-cov==2.12.1
RUN pip install sphinx==3.5.4
RUN pip install sphinx-autodoc-typehints==1.12.0
RUN pip install sphinx-book-theme==0.0.42

# Instalar módulos de ABIDES desde el submódulo
RUN cd abides-jpmc-public/abides-core && python3 setup.py install && cd ../..
RUN cd abides-jpmc-public/abides-markets && python3 setup.py install && cd ../..
RUN cd abides-jpmc-public/abides-gym && python3 setup.py install && cd ../..

# Comando por defecto al arrancar el contenedor
CMD ["/bin/bash"]
