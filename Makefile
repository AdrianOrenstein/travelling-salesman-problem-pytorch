SHELL=/bin/bash
CONDAROOT = ~/miniconda3

init:
		@echo "Creating local conda env from yaml..."
		conda init bash && \
		source ~/miniconda3/etc/profile.d/conda.sh && \
		git clone https://github.com/jvkersch/pyconcorde && \
		cd pyconcorde && \
		pip install -e . && \
		cd .. && \
		conda env create -f environment.yaml --prefix ./envs

init_fresh:
		@echo "Creating local conda env from scratch..."
		conda init bash && \
		source ~/miniconda3/etc/profile.d/conda.sh && \
		conda create --prefix ./envs -y python=3.8 numpy pandas matplotlib scikit-learn tqdm requests pylint black && \
		conda activate ./envs && \
		conda install -yc pytorch pytorch torchvision cudatoolkit=10.1 && \
		conda install -yc conda-forge pytorch-lightning tensorboard && \
		conda install -yc conda-forge pytorch_geometric && \
		conda install -yc anaconda cython && \
		git clone https://github.com/jvkersch/pyconcorde && \
		cd pyconcorde && \
		pip install -e . && \
		cd .. && \
		conda env export | grep -v "^prefix: " > environment.yaml

clean:
		@echo "Cleaning up..."
		rm -rf ./envs