build:
	export DOCKER_BUILDKIT=1
	docker build --secret id=env,src=.env --platform linux/amd64 -t "ao_arc_app" .
	docker tag ao_arc_app aolabs/arc-agi

run:
	docker run -p 8501:8501 aolabs/arc-agi:latest

push:
	docker push aolabs/arc-agi:latest
