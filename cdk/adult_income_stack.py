import os
import json
import boto3
import aws_cdk as cdk
from constructs import Construct

from aws_cdk import (
    Stack,
    RemovalPolicy,
    CfnOutput,
    aws_s3 as s3,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_sagemaker as sagemaker,
    aws_secretsmanager as secretsmanager,
)


def resource_exists(client, check_fn):
    """Helper to check if a resource exists."""
    try:
        check_fn(client)
        return True
    except Exception:
        return False


class AdultIncomeSageMakerStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        github_repo: str,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        account  = self.account
        region   = self.region
        aws_region = os.environ.get("AWS_REGION", "us-east-1")

        # ─── Boto3 Clients ───────────────────────────────────
        ec2_client = boto3.client("ec2",        region_name=aws_region)
        iam_client = boto3.client("iam",        region_name=aws_region)
        sm_client  = boto3.client("sagemaker",  region_name=aws_region)
        s3_client  = boto3.client("s3",         region_name=aws_region)

        # ─── VPC + Subnets ───────────────────────────────────
        vpcs = ec2_client.describe_vpcs(
            Filters=[{"Name": "isDefault", "Values": ["true"]}]
        )
        vpc_id = vpcs["Vpcs"][0]["VpcId"]
        subnets = ec2_client.describe_subnets(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
        )
        subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]][:2]
        print(f"✅ VPC ID   : {vpc_id}")
        print(f"✅ Subnets  : {subnet_ids}")

        # ─── 1. S3 Bucket — Import if Exists ─────────────────
        bucket_name = f"sagemaker-adult-income-cdk-{account}"
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ S3 Bucket exists — importing: {bucket_name}")
            bucket = s3.Bucket.from_bucket_name(
                self, "TrainingBucket",
                bucket_name
            )
        except Exception:
            print(f"⏳ S3 Bucket not found — creating: {bucket_name}")
            bucket = s3.Bucket(
                self, "TrainingBucket",
                bucket_name=bucket_name,
                removal_policy=RemovalPolicy.RETAIN,
                block_public_access=s3.BlockPublicAccess.BLOCK_ALL
            )

        # ─── 2. SageMaker Execution Role — Import if Exists ──
        sagemaker_role_name = "AdultIncomeSageMakerExecutionRole"
        try:
            existing_role = iam_client.get_role(RoleName=sagemaker_role_name)
            role_arn = existing_role["Role"]["Arn"]
            print(f"✅ SageMaker Role exists — importing: {sagemaker_role_name}")
            sagemaker_role = iam.Role.from_role_arn(
                self, "SageMakerExecutionRole",
                role_arn
            )
        except iam_client.exceptions.NoSuchEntityException:
            print(f"⏳ SageMaker Role not found — creating: {sagemaker_role_name}")
            sagemaker_role = iam.Role(
                self, "SageMakerExecutionRole",
                role_name=sagemaker_role_name,
                assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
                description="SageMaker execution role for Adult Income pipeline",
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "AmazonSageMakerFullAccess"
                    ),
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "AmazonS3FullAccess"
                    )
                ]
            )

        # ─── 3. SageMaker Studio Domain — Import if Exists ───
        domain_name = "adult-income-sagemaker-studio-cdk"
        existing_domains = sm_client.list_domains()
        existing_domain  = next(
            (d for d in existing_domains["Domains"]
             if d["DomainName"] == domain_name),
            None
        )

        if existing_domain:
            domain_id = existing_domain["DomainId"]
            print(f"✅ Studio Domain exists — importing: {domain_name} ({domain_id})")
            # Reference domain ID directly (no CDK construct needed)
            domain_id_value = domain_id
        else:
            print(f"⏳ Studio Domain not found — creating: {domain_name}")
            domain = sagemaker.CfnDomain(
                self, "StudioDomain",
                domain_name=domain_name,
                auth_mode="IAM",
                default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                    execution_role=sagemaker_role.role_arn,
                    jupyter_server_app_settings=sagemaker.CfnDomain\
                        .JupyterServerAppSettingsProperty(
                            default_resource_spec=sagemaker.CfnDomain\
                                .ResourceSpecProperty(
                                    instance_type="system"
                                )
                        ),
                    kernel_gateway_app_settings=sagemaker.CfnDomain\
                        .KernelGatewayAppSettingsProperty(
                            default_resource_spec=sagemaker.CfnDomain\
                                .ResourceSpecProperty(
                                    instance_type="ml.t3.medium"
                                )
                        )
                ),
                subnet_ids=subnet_ids,
                vpc_id=vpc_id,
                tags=[
                    cdk.CfnTag(key="Project",     value="Adult-Income-Pipeline"),
                    cdk.CfnTag(key="Environment", value="dev"),
                    cdk.CfnTag(key="ManagedBy",   value="CDK")
                ]
            )
            domain_id_value = domain.attr_domain_id

        # ─── 4. User Profile — Create Only if Domain is New ──
        if not existing_domain:
            user_profile = sagemaker.CfnUserProfile(
                self, "StudioUserProfile",
                domain_id=domain_id_value,
                user_profile_name="adult-income-user",
                user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                    execution_role=sagemaker_role.role_arn
                )
            )
            user_profile.node.add_dependency(domain)

        # ─── 5. GitHub Token Secret Reference ────────────────
        github_token_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "GitHubTokenSecret",
            "github-token"
        )

        # ─── 6. Lambda Role — Import if Exists ───────────────
        lambda_role_name = "LambdaAdultDeployTriggerRole"
        try:
            existing_lambda_role = iam_client.get_role(RoleName=lambda_role_name)
            lambda_role_arn = existing_lambda_role["Role"]["Arn"]
            print(f"✅ Lambda Role exists — importing: {lambda_role_name}")
            lambda_role = iam.Role.from_role_arn(
                self, "LambdaDeployTriggerRole",
                lambda_role_arn
            )
        except iam_client.exceptions.NoSuchEntityException:
            print(f"⏳ Lambda Role not found — creating: {lambda_role_name}")
            lambda_role = iam.Role(
                self, "LambdaDeployTriggerRole",
                role_name=lambda_role_name,
                assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                description="Lambda role to trigger GitHub Actions (Adult Income)",
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name(
                        "service-role/AWSLambdaBasicExecutionRole"
                    )
                ]
            )
            github_token_secret.grant_read(lambda_role)

        # ─── 7. Lambda Function — Import if Exists ───────────
        lambda_name = "trigger-adult-deploy-pipeline"
        lambda_client = boto3.client("lambda", region_name=aws_region)

        try:
            existing_fn = lambda_client.get_function(FunctionName=lambda_name)
            lambda_arn  = existing_fn["Configuration"]["FunctionArn"]
            print(f"✅ Lambda exists — importing: {lambda_name}")
            deploy_trigger_fn = lambda_.Function.from_function_arn(
                self, "DeployTriggerLambda",
                lambda_arn
            )
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"⏳ Lambda not found — creating: {lambda_name}")
            lambda_code = f"""
import json
import urllib.request
import boto3

def lambda_handler(event, context):
    print(f"Event received: {{json.dumps(event)}}")
    detail = event.get("detail", {{}})
    status = detail.get("ModelApprovalStatus", "")
    print(f"Model Approval Status: {{status}}")

    if status != "Approved":
        print("Model not approved - skipping deploy")
        return {{"statusCode": 200, "body": "Not approved"}}

    sm_client = boto3.client("secretsmanager")
    secret    = sm_client.get_secret_value(SecretId="github-token")
    token     = secret["SecretString"]

    repo = "{github_repo}"
    url  = f"https://api.github.com/repos/{{repo}}/actions/workflows/deploy-pipeline.yml/dispatches"
    payload = json.dumps({{"ref": "master"}}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={{
            "Authorization": f"Bearer {{token}}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            print(f"Pipeline 3 triggered! Status: {{response.status}}")
            return {{"statusCode": 200, "body": "Pipeline 3 triggered!"}}
    except Exception as e:
        print(f"Failed to trigger: {{str(e)}}")
        raise
"""
            deploy_trigger_fn = lambda_.Function(
                self, "DeployTriggerLambda",
                function_name=lambda_name,
                runtime=lambda_.Runtime.PYTHON_3_10,
                handler="index.lambda_handler",
                code=lambda_.Code.from_inline(lambda_code),
                role=lambda_role,
                timeout=cdk.Duration.seconds(30),
                description="Triggers GitHub Actions Pipeline 3 on Adult Income model approval"
            )

        # ─── 8. EventBridge Rule — Always Upsert ─────────────
        model_approved_rule = events.Rule(
            self, "ModelApprovedRule",
            rule_name="adult-model-approved-rule",
            description="Trigger deploy pipeline when Adult Income model approved",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Package State Change"],
                detail={
                    "ModelApprovalStatus": ["Approved"],
                    "ModelPackageGroupName": ["AdultIncomeGroup"]
                }
            )
        )
        model_approved_rule.add_target(
            targets.LambdaFunction(deploy_trigger_fn)
        )

        # ─── 9. CloudFormation Outputs ────────────────────────
        CfnOutput(
            self, "DomainId",
            value=domain_id_value if isinstance(domain_id_value, str)
                  else domain_id_value,
            description="SageMaker Studio Domain ID",
            export_name="AdultIncomeDomainId"
        )
        CfnOutput(
            self, "RoleArn",
            value=sagemaker_role.role_arn,
            description="SageMaker Execution Role ARN",
            export_name="AdultIncomeRoleArn"
        )
        CfnOutput(
            self, "BucketName",
            value=bucket.bucket_name,
            description="Training S3 Bucket Name",
            export_name="AdultIncomeBucketName"
        )
        CfnOutput(
            self, "LambdaArn",
            value=deploy_trigger_fn.function_arn,
            description="Deploy Trigger Lambda ARN",
            export_name="AdultIncomeLambdaArn"
        )