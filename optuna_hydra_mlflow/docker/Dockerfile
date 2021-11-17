FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


ARG USER

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install nano wget curl

RUN pip3 install jupyter click numpy matplotlib seaborn pandas tqdm
RUN pip3 install torch torchvision
RUN pip3 install hydra-core mlflow 
RUN pip3 install hydra-optuna-sweeper

RUN useradd -m -s /bin/bash ${USER}

USER ${USER}

RUN jupyter notebook --generate-config \
    && sed -i.back \
    -e "s:^#c.NotebookApp.token = .*$:c.NotebookApp.token = u'':" \
    -e "s:^#c.NotebookApp.ip = .*$:c.NotebookApp.ip = '0.0.0.0':" \
    -e "s:^#c.NotebookApp.open_browser = .*$:c.NotebookApp.open_browser = False:" \
    /home/${USER}/.jupyter/jupyter_notebook_config.py

WORKDIR /home/${USER}
