vhls=/scratch/jc9016/tools/Xilinx/2020.2
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)

# Make sure the repo is up to date
sync:
	git submodule sync
	git submodule update --init --recursive
	bash mlir-air/utils/clone-llvm.sh 
	bash mlir-air/utils/clone-mlir-aie.sh 

# Build Docker container
build-docker: 
	(cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) . --tag mase-ubuntu2204)

shell: build-docker
	docker run -it --shm-size 256m --hostname mase-ubuntu2204 -u $(user) -v $(vhls):$(vhls) -v $(shell pwd):/workspace mase-ubuntu2204:latest /bin/bash 

build:
	bash scripts/build-llvm.sh
	bash scripts/build-mase-hls.sh
	bash scripts/build-aie.sh
	bash scripts/build-air.sh

clean:
	rm -rf llvm/build
