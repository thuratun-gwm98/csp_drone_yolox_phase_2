.PHONY: docker-build docker-run-dev test

docker-build:
	
	docker build -t yolox -f YOLOX.Dockerfile .

docker-run:
	docker run -it --rm --gpus all -v ./:/home/yolox:rw -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ --ipc=host --network=host yolox