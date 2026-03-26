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

# Todas las dependencias en una sola capa
RUN pip install --no-cache-dir \
    coloredlogs==15.0.1 \
    numpy==1.22.0 \
    pandas==1.2.4 \
    pomegranate==0.14.6 \
    psutil==5.8.0 \
    scipy==1.10.0 \
    termcolor \
    "ray[rllib]==1.13.0" \
    p_tqdm==1.3.3 \
    pre-commit==2.13 \
    pytest==6.2.4 \
    pytest-cov==2.12.1 \
    sphinx==3.5.4 \
    sphinx-autodoc-typehints==1.12.0 \
    sphinx-book-theme==0.0.42

# Instalar módulos de ABIDES
RUN cd abides-jpmc-public/abides-core && python3 setup.py install && \
    cd ../abides-markets && python3 setup.py install && \
    cd ../abides-gym && python3 setup.py install

# Comando por defecto al arrancar el contenedor
# Alias para probar ABIDES fácilmente
RUN echo 'alias test-abides="abides abides-jpmc-public/abides-markets/abides_markets/configs/rmsc04.py --end_time 10:00:00"' >> ~/.bashrc
CMD ["/bin/bash"]