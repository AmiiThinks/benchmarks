# Note: this Dockerfile expects that Atari ROMs retrieved following the
# instructions from atari-py: https://github.com/openai/atari-py#roms.
# It should specify a directory (e.g. ~/roms) that contains ROMS.rar.
# It should be run from the rom directory.

ARG base_image=dopamine/core
FROM ${base_image}

# Copy ROMs into the image.
RUN mkdir /root/roms

RUN apt-get install rar unzip -y
RUN pip install atari_py ale-py

COPY ./Roms.rar /root/roms/
RUN rar x /root/roms/Roms.rar /root/roms/

RUN python -m atari_py.import_roms /root/roms
RUN ale-import-roms /root/roms/ROMS

COPY ./configs /configs
