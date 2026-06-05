.PHONY: build pipeline dashboard test shell clean spark-up spark-etl spark-down spark-partition spark-pipeline

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

WORKERS ?= 1
DATANODES ?= 1

spark-up:         ## Start HDFS + Spark; scale with WORKERS=3 DATANODES=3
	WORKERS=$(WORKERS) DATANODES=$(DATANODES) docker compose --profile spark up -d

spark-partition:  ## Build the division-partitioned warehouse in HDFS
	docker compose --profile spark --profile spark-jobs run --rm --no-deps spark-partition

spark-pipeline:   ## Distributed 5-phase MLlib run; scope with DIVISIONS=Khulna,Barisal
	docker compose --profile spark --profile spark-jobs run --rm --no-deps -e DIVISIONS=$(DIVISIONS) spark-pipeline

spark-etl:        ## Run the distributed ETL job on the Spark cluster
	docker compose --profile spark --profile spark-jobs run --rm --no-deps spark-etl

spark-down:       ## Stop the Spark cluster
	docker compose --profile spark down

clean:            ## Remove generated artifacts (keeps raw data)
	rm -rf data/processed models reports
