.PHONY: requirements

PROJECT_NAME=shopee

create_environment:
	conda create --yes -n $(PROJECT_NAME) -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids-blazing=0.18 python=3.7 cudatoolkit=10.2
	# conda create --yes --name $(PROJECT_NAME) python=3.7 anaconda

requirements:
	pip install -r requirements.txt
	conda install --yes ipykernel
	python -m ipykernel install --user --name $(PROJECT_NAME) --display-name "$(PROJECT_NAME)"
