vhls=/mnt/applications/Xilinx/23.1
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)

# Make sure the repo is up to date
sync:
	git submodule sync
	git submodule update --init --recursive

# Only needed if you are using the MLIR flow - it will be slow!
sync-mlir:
	bash mlir-air/utils/github-clone-build-libxaie.sh
	bash mlir-air/utils/clone-llvm.sh 
	bash mlir-air/utils/clone-mlir-aie.sh 

sync-fpgaconvnet:
	git submodule sync
	git submodule update --init --recursive "machop/third-party/fpgaconvnet-optimiser"

# Build Docker container
build-docker: 
	docker build --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) -f Docker/Dockerfile --tag mase-ubuntu2204 Docker

shell: build-docker
	docker run -it --shm-size 256m --hostname mase-ubuntu2204 -w /workspace -v $(shell pwd):/workspace:z mase-ubuntu2204:latest /bin/bash 

# There is a historical reason that test files are stored under the current directory
# Short-term solution: call scripts under /tmp so we can clean it properly
test-hw:
	mkdir -p ./tmp
	(cd tmp; python3 ../scripts/test-hardware.py -a || exit 1)

test-sw:
	mkdir -p ./tmp
	(cd tmp; bash ../scripts/test-machop.sh || exit 1)

test-all: test-hw test-sw
	mkdir -p ./tmp
	(cd tmp; python3 ../scripts/test-torch-mlir.py || exit 1)

build:
	bash scripts/build-llvm.sh || exit 1
	bash scripts/build-mase-hls.sh || exit 1

build-aie:
	bash scripts/build-aie.sh || exit 1
	bash scripts/build-air.sh || exit 1

clean:
	rm -rf llvm
	rm -rf aienginev2 mlir-air/build mlir-aie
	rm -rf hls/build
	rm -rf vck190_air_sysroot
	rm -rf tmp mase_output
