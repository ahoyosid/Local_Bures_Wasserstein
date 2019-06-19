ENV_NAME=lbw
install:
	conda env create -f environment.yml &&\
	source ${HOME}/anaconda3/bin/activate $(ENV_NAME) &&\
	conda install pytorch torchvision -c pytorch &&\
	conda install -c anaconda scikit-image &&\
	conda install -c conda-forge pot &&\
	source ${HOME}/anaconda3/bin/deactivate 

run_shapes:
	source ${HOME}/anaconda3/bin/activate $(ENV_NAME)  &&\
	python example_shape_barycenter.py &&\
	source ${HOME}/anaconda3/bin/deactivate

run_classifier:
	source ${HOME}/anaconda3/bin/activate $(ENV_NAME)  &&\
	python example_mixing_groups.py &&\
	source ${HOME}/anaconda3/bin/deactivate  
