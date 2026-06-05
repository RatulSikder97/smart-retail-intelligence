.PHONY: build pipeline dashboard test shell clean

build:            ## Build the Docker image
	docker compose build

pipeline:         ## Run the full ML pipeline (phases 1-5) inside Docker
	docker compose run --rm pipeline

dashboard:        ## Serve the Streamlit dashboard on http://localhost:8501
	docker compose up dashboard

test:             ## Run the test suite inside Docker
	docker compose run --rm test

shell:            ## Interactive shell inside the container
	docker compose run --rm --entrypoint bash pipeline

clean:            ## Remove generated artifacts (keeps raw data)
	rm -rf data/processed models reports
