#!/usr/bin/env python3
import aws_cdk as cdk
import boto3
import os

from adult_income_stack import AdultIncomeSageMakerStack

app = cdk.App()

# Get account and region from environment
account = boto3.client("sts").get_caller_identity()["Account"]
region  = os.environ.get("AWS_REGION", "us-east-1")

AdultIncomeSageMakerStack(
    app,
    "AdultIncomeSageMakerStack",
    env=cdk.Environment(account=account, region=region),
    github_repo="kabdul2010/aws-mlops-e2e"
)

app.synth()
