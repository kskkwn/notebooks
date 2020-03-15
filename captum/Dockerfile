FROM pytorch/pytorch
ARG USERNAME

RUN pip install captum jupyter jupyterthemes
RUN pip install matplotlib click
RUN pip install flask
RUN pip install pillow


RUN useradd -m -s /bin/bash ${USERNAME}

USER ${USERNAME}

RUN jupyter notebook --generate-config \
 && sed -i.back \
    -e "s:^#c.NotebookApp.token = .*$:c.NotebookApp.token = u'':" \
    -e "s:^#c.NotebookApp.ip = .*$:c.NotebookApp.ip = '0.0.0.0':" \
    -e "s:^#c.NotebookApp.open_browser = .*$:c.NotebookApp.open_browser = False:" \
    /home/${USERNAME}/.jupyter/jupyter_notebook_config.py

RUN jt -t gruvboxd -T -N -cellw 80%


WORKDIR /home/${USERNAME}
