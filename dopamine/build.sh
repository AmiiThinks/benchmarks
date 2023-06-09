if [ -z "$(ls -A submodule)" ]; then
    git submodule init
    git submodule update
fi

if [ ! -f ./Roms.rar ]; then
    wget http://www.atarimania.com/roms/Roms.rar -O ./Roms.rar
fi

docker build -f core/Dockerfile -t dopamine/core .
docker build -f atari/Dockerfile -t dopamine/atari .

time docker run --privileged --gpus all --name rainbow_atari -it dopamine/atari  python -um dopamine.discrete_domains.train --base_dir /tmp --gin_files /configs/rainbow.gin
