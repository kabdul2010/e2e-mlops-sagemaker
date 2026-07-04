# e2e MLOps test 04/07/2026
import boto3
import sagemaker
import os
import json
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile


def run_pipeline():

    # ─── Setup ──────────────────────────────────────────────
    region  = os.environ.get("AWS_REGION", "us-east-1")
    account = boto3.client("sts", region_name=region)\
                   .get_caller_identity()["Account"]
    bucket  = f"sagemaker-adult-income-cdk-{account}"

    boto_session = boto3.Session(region_name=region)
    sm_client    = boto3.client("sagemaker", region_name=region)
    s3_client    = boto3.client("s3", region_name=region)

    # ─── Create Bucket if Not Exists ────────────────────────
    try:
        s3_client.create_bucket(Bucket=bucket)
        print(f"✅ Bucket created : {bucket}")
    except Exception:
        print(f"✅ Bucket exists  : {bucket}")

    # ─── ✅ UPLOAD DATA TO S3 VIA CODE (no manual upload) ───
    print("⏳ Uploading dataset from repo to S3...")
    local_data_path = "data/adult_final.csv"
    s3_client.upload_file(local_data_path, bucket, "data/adult_final.csv")
    print(f"✅ Data uploaded to s3://{bucket}/data/adult_final.csv")

    # ─── Read Infra Outputs from S3 ─────────────────────────
    try:
        response  = s3_client.get_object(
            Bucket=bucket,
            Key="infra/infra_outputs.json"
        )
        infra     = json.loads(response["Body"].read())
        role      = infra["role_arn"]
        domain_id = infra["domain_id"]
        bucket    = infra["bucket_name"]

        print(f"✅ Infra outputs loaded")
        print(f"✅ Domain ID : {domain_id}")
        print(f"✅ Role ARN  : {role}")
        print(f"✅ Bucket    : {bucket}")

    except Exception as e:
        print(f"⚠️ S3 read failed : {str(e)}")
        role      = os.environ.get("AWS_ROLE_ARN")
        domain_id = os.environ.get("DOMAIN_ID", "")
        print(f"✅ Fallback Role   : {role}")

    session = sagemaker.Session(
        boto_session=boto_session,
        sagemaker_client=sm_client,
        default_bucket=bucket
    )

    # ─── Step 1: Data Preparation ───────────────────────────
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session
    )

    data_prep_step = ProcessingStep(
        name="DataPreparation",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{bucket}/data/adult_final.csv",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/processed/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/processed/test"
            )
        ],
        code="pipeline/preprocessing.py"
    )

    # ─── Step 2: Model Training ─────────────────────────────
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="pipeline",
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        use_spot_instances=False,
        max_run=3600,
        sagemaker_session=session,
        hyperparameters={
            "n_estimators": 100,
            "random_state": 42
        },
        metric_definitions=[
            {"Name": "accuracy",  "Regex": "accuracy: ([0-9\\.]+)"},
            {"Name": "precision", "Regex": "precision: ([0-9\\.]+)"},
            {"Name": "recall",    "Regex": "recall: ([0-9\\.]+)"},
            {"Name": "f1",        "Regex": "f1: ([0-9\\.]+)"}
        ]
    )

    train_step = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=data_prep_step.properties\
                        .ProcessingOutputConfig\
                        .Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # ─── Step 3: Model Evaluation ───────────────────────────
    evaluator = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        sagemaker_session=session
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    eval_step = ProcessingStep(
        name="ModelEvaluation",
        processor=evaluator,
        inputs=[
            ProcessingInput(
                source=train_step.properties\
                       .ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=data_prep_step.properties\
                       .ProcessingOutputConfig\
                       .Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/evaluation"
            )
        ],
        code="pipeline/evaluate.py",
        property_files=[evaluation_report]
    )

    # ─── Step 4: Model Registration ─────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{bucket}/evaluation/evaluation.json",
            content_type="application/json"
        )
    )

    register_step = RegisterModel(
        name="ModelRegistration",
        estimator=estimator,
        model_data=train_step.properties\
                  .ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="AdultIncomeGroup",
        approval_status="PendingManualApproval",
        model_metrics=model_metrics
    )

    # ─── Step 5: Quality Gate ───────────────────────────────
    accuracy_condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value"
        ),
        right=0.75
    )

    condition_step = ConditionStep(
        name="QualityGate",
        conditions=[accuracy_condition],
        if_steps=[register_step],
        else_steps=[]
    )

    # ─── Build & Run Pipeline ───────────────────────────────
    pipeline = Pipeline(
        name="AdultIncomePredictionPipeline",
        steps=[
            data_prep_step,
            train_step,
            eval_step,
            condition_step
        ],
        sagemaker_session=session
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()

    print(f"✅ Pipeline Started!")
    print(f"✅ Execution ARN : {execution.arn}")
    print(f"✅ Monitor here  : SageMaker → Pipelines → AdultIncomePredictionPipeline")


if __name__ == "__main__":
    run_pipeline()
